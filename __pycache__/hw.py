import random
k = []
import numpy as np
m = Model()
results = []

last_fp = 1.2
prev_fp = 0.4

results = []
repetition = 10
results = []

def run_exp(last_trial_t, prev_last_trial_t, max_trials,rep=10):
    fps = [0.4]*100
    fps.extend([0.8]*100)
    fps.extend([1.2]*100)
    fps.extend([1.6]*100)
    results = []
    for r in range(rep):
        m = Model()
        m.le = 1.0
        m.time += .05 
        
        for trial_no in range(max_trials): 
         
            m.time += random.uniform(1.0,1.5)
            fp = random.choice(fps)
            prep_adv = 0
            if trial_no == max_trials - 2 :
                fp = prev_last_trial_t
            if trial_no == max_trials - 1:
                fp = last_trial_t

            
            if trial_no > 0:
                request = Chunk(name = "request", slots = {"isa":"fp-time-fact"})
                chunk, latency = m.retrieve(request)
                m.time += latency
                
                if chunk != None:
                    #m.add_encounter(chunk)
                    m.time += 0.05
                    exp_fp = pulses_to_time(chunk.slots['pulses'])
                    if trial_no == max_trials - 1:
                            rt = 0.395
                            if exp_fp < fp: 
                                 rt = 0.395 - min(0.055,abs(fp-exp_fp))
                                 #print( min(1.2,abs(fp-exp_fp) ))
                                 
                            results.append(rt)
                
            
            pulses = time_to_pulses(fp)
            fact = Chunk(name = "fp_observed:%d"%trial_no, slots = {"isa":"fp-time-fact", "pulses": pulses})
            m.add_encounter(fact)
            m.time += 0.05
            #m.time += 0.3
    return np.mean(results)

##0.4 fp_n-1
#fp: 0.4 : 0.3602234325610658
#fp: 0.8: 0.34733755575854003
#fp: 1.2: 0.33733755575854003
######################
##0.8  fp_n-1 
#fp: 0.4 : 0.3275
#fp: 0.8: 0.
#fp: 1.2: 0
print("fp n-1 : 0.4")
arr=[]
x = [0.4,0.8,1.2,1.6]
n_trials = 900
rep = 50
rt_a_1 = run_exp(0.4, 0.4, n_trials ,rep=rep)
rt_a_2 = run_exp(0.8, 0.4, n_trials ,rep=rep)
rt_a_3 = run_exp(1.2, 0.4, n_trials ,rep=rep)
rt_a_4 = run_exp(1.6, 0.4, n_trials ,rep=rep)
print([rt_a_1,rt_a_2,rt_a_3,rt_a_4])
arr.extend([rt_a_1,rt_a_2,rt_a_3,rt_a_4])
plt.xlabel('Foreperiod')
plt.ylabel('Reaction Time')
plt.plot(x ,[rt_a_1,rt_a_2,rt_a_3,rt_a_4],marker='o',color='red') 


#plt.suptitle(subtitle)
rt_b_1 = run_exp(0.4, 0.8, n_trials ,rep=rep)
rt_b_2 = run_exp(0.8, 0.8, n_trials ,rep=rep)
rt_b_3 = run_exp(1.2, 0.8, n_trials ,rep=rep)
rt_b_4 = run_exp(1.6, 0.8, n_trials ,rep=rep)

print("fp n-1 : 0.8")
print([rt_b_1,rt_b_2,rt_b_3,rt_b_4])
plt.plot(x ,[rt_b_1,rt_b_2,rt_b_3,rt_b_4],marker='o',color='green') 
arr.extend([rt_b_1,rt_b_2,rt_b_3,rt_b_4])


rt_c_1 = run_exp(0.4, 1.2, n_trials ,rep=rep)
rt_c_2 = run_exp(0.8, 1.2, n_trials ,rep=rep)
rt_c_3 = run_exp(1.2, 1.2, n_trials ,rep=rep)
rt_c_4 = run_exp(1.6, 1.2, n_trials ,rep=rep)
print("fp n-1 : 1.2")
print([rt_c_1,rt_c_2,rt_c_3,rt_c_4])

plt.plot(x ,[rt_c_1,rt_c_2,rt_c_3,rt_c_4],marker='o',color='blue') 
print("fp n-1 : 1.2")
arr.extend([rt_c_1,rt_c_2,rt_c_3,rt_c_4])

rt_d_1 = run_exp(0.4, 1.6, n_trials ,rep=rep)
rt_d_2 = run_exp(0.8, 1.6, n_trials ,rep=rep)
rt_d_3 = run_exp(1.2, 1.6, n_trials ,rep=rep)
rt_d_4 = run_exp(1.6, 1.6, n_trials ,rep=rep)
arr.extend([rt_d_1,rt_d_2,rt_d_3,rt_d_4])
plt.ylim(min(arr)-0.01,max(arr)+0.01)

print([rt_d_1,rt_d_2,rt_d_3,rt_d_4])
plt.plot(x, [rt_d_1,rt_d_2,rt_d_3,rt_d_4], marker='o',color='purple')
plt.show()
print("min %f"%min(arr))