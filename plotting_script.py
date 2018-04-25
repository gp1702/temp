import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

##loading the npy files

q_learning=np.load('All_rewards_Q_learning.npy')

#taking the mean of 5 runs
avg_rewards=np.mean(q_learning,axis=0)

##get the std of the data
std_rewards=np.std(q_learning,axis=0)


smoothing_window=5

#avg_rewards=pd.DataFrame(avg_rewards)
smoothed_avg_rewards=pd.Series(avg_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(smoothed_avg_rewards,color='red')

##filling in between stds

plt.fill_between(range(len(smoothed_avg_rewards)),smoothed_avg_rewards+std_rewards,smoothed_avg_rewards-std_rewards,alpha=0.2,edgecolor='red',facecolor='red')

plt.xlabel('Number of episodes')
plt.ylabel('Average Cumulative Reward')
plt.title('Q_learning in HIV Drug Scheduling')
plt.show()