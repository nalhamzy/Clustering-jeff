"""
This is a more complex example on performing clustering on large scale dataset.
This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.
A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.
The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).
In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time
import pandas as pd
import streamlit as st




# Model used for computing sentence embeddings. 
@st.cache(persist=True, allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model 


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')



def main():
    df = []
    ### please select a file to load ####
    file_path = 'vision boards.xlsx'
    threshold = st.slider('Similarity Threshold:',0.0, 1.0,0.95,0.05)
    min_community_size = st.slider('Minimum size of a cluster:',1, 10,2,1)
    max_size = 50
    uploaded_file = st.file_uploader("Upload Files",type=['csv','xlsx'])



    sentences = st.text_area("Paste sentences here:", height=300)

    if sentences != None:
        sentences = sentences.splitlines()
        df = pd.DataFrame(columns=['Questions'])
        for sentence in sentences:
            df = df.append({'Questions':sentence},ignore_index=True)

    if uploaded_file is not None:
            if "csv" in uploaded_file.name:
                df = pd.read_csv(uploaded_file)
            elif "xlsx" in uploaded_file.name:
                df = pd.read_excel(uploaded_file)
            st.write(df)
    if uploaded_file == None and sentences == None or len(sentences) < 1:
        return 
    
    
    
    ## remove null records 
    
    df.dropna()
    model = load_model()
    corpus_sentences = set()

    ### read the questions from the dataframe ###
    ## please change the column name 'Questions' -> The column name in your file 
    column = df.columns[0]
    for idx, row in df.iterrows():
        
        if type(row[column]) == type(''):
            corpus_sentences.add(row[column])

    ### lists unique sentences ###
    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=1, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")
    start_time = time.time()

    #Two parameters to tune:
    #min_cluster_size: Only consider cluster that have at least 2 elements
    #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar

    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold, init_max_size=max_size)

    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    ## function used to extract the shortest sentence from the given cluster ##
    def get_shortest_title_cluster(cluster, corpus_sentences):
        title = corpus_sentences[cluster[0]]
        for idx in cluster:
            if len(corpus_sentences[idx]) < len(title):
                title = corpus_sentences[idx]
        return title

    ### prepare a new dataframe to store the results ###
    clusters_df = pd.DataFrame(columns=['Cluster','Question'])
    num_clusters = 0
    for i, cluster in enumerate(clusters):
        cluster_name = "Cluster {}".format(i)
        title = get_shortest_title_cluster(cluster, corpus_sentences)
        print("cluster: {title}".format(title = title))
        for sentence_id in cluster:
            clusters_df = clusters_df.append({'Cluster':title, 'Question':corpus_sentences[sentence_id]},ignore_index=True)
        num_clusters += 1
    missing_questions_title = 'unspecified cluster'
    for s in corpus_sentences:
         res = clusters_df.loc[clusters_df['Question'] == s]
         if len(res) < 1: 
             clusters_df = clusters_df.append({'Cluster':missing_questions_title, 'Question':s},ignore_index=True)

    st.text('# Clusters: {num_clusters}'.format(num_clusters=len(clusters_df['Cluster'].unique())))

    clusters_df
    csv = convert_df(clusters_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='clusters.csv',
        mime='text/csv',
    )
if __name__ == '__main__':
	main()

