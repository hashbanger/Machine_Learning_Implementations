#implementing the UCB for multi armed bandits problem
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset for simulations
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Implementing the algorithm
import math
N = 10000               #No. of Rounds
d = 10                  #No. of bandits
ads_selected = []
number_of_selections = [0]*d
sum_of_rewards = [0]*d
total_reward = 0
for n in range(0,N):
    #reward = 0
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if(number_of_selections[i] > 0):
            average_rewards = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = delta_i + average_rewards
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1 
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualising the results
plt.hist(ads_selected)
plt.xlabel('Ads')
plt.ylabel('No of Selections')
plt.grid()
plt.show()
