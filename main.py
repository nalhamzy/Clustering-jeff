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
corpus_sentences = set()
import pandas as pd
import streamlit as st

# Model used for computing sentence embeddings. 
model = SentenceTransformer('all-MiniLM-L6-v2')


### please select a file to load ####
file_path = 'vision boards.xlsx'
uploaded_file = st.file_uploader("Upload Files",type=['xlsx','csv'])
if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

## remove null records 
df.dropna()

### read the questions from the dataframe ###
## please change the column name 'Questions' -> The column name in your file 
for idx, row in df.iterrows():
    if type(row['Questions']) == type(''):
      corpus_sentences.add(row['Questions'])

### lists unique sentences ###
corpus_sentences = list(corpus_sentences)
print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=1, show_progress_bar=True, convert_to_tensor=True)

print(ccffffc)
print("Start clustering")
start_time = time.time()

#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 2 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar

clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.95, init_max_size=50)

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
for i, cluster in enumerate(clusters):
    cluster_name = "Cluster {}".format(i)
    title = get_shortest_title_cluster(cluster, corpus_sentences)
    print("cluster: {title}".format(title = title))
    for sentence_id in cluster:
        clusters_df = clusters_df.append({'Cluster':title, 'Question':corpus_sentences[sentence_id]},ignore_index=True)

clusters_df