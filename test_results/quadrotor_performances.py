import numpy as np


def calculate_mean_std(list):
    # Combine the lists into a single array
    combined_data = np.concatenate(list)
    
    # Calculate mean and standard deviation
    mean = np.mean(combined_data)
    std_dev = np.std(combined_data)
    print(f"========== mean = {mean}, std_dev = {std_dev} ========== \n")
    
    # return mean, std_dev


### Data loading: method + method random seed + test env, eg, ours_42_randomhj
## quadrotor_randomhj
# ours in randomhj
ours_42_randomhj = [1000, 1000, 1000, 1000, 12, 2, 1000, 1000, 4, 1000] 
ours_2024_randomhj = [4, 1000, 15, 1000, 1000, 5, 1000, 1000, 1000, 2] 
ours_40226_randomhj = [1000, 1000, 1000, 1000, 2, 24, 1000, 1000, 1000, 1000] 

# pure ppo in randomhj
ppo_42_randomhj = [3, 1000, 29, 49, 15, 4, 1000, 119, 5, 89] 
ppo_2024_randomhj = [7, 42, 22, 1000, 47, 5, 37, 1000, 29, 4] 
ppo_40226_randomhj = [12, 1000, 1000, 58, 3, 21, 17, 1000, 1000, 13]

# rarl in randomhj
rarl_42_randomhj =  [7, 1000, 56, 26, 8, 4, 1000, 1000, 5, 77] 
rarl_2024_randomhj = [6, 68, 6, 1000, 23, 3, 1000, 1000, 28, 3] 
rarl_422_randomhj = [13, 21, 18, 28, 13, 5, 13, 15, 6, 8]
rarl_20242_randomhj = [9, 34, 20, 11, 74, 6, 22, 26, 22, 3]

# rap in randomhj
rap_42_randomhj = [49, 33, 22, 59, 13, 4, 1000, 1000, 5, 218] 
rap_2024_randomhj = [5, 34, 22, 1000, 20, 5, 80, 1000, 24, 4]
rap_422_randomhj = [13, 1000, 25, 17, 5, 3, 20, 22, 3, 29]

## quadrotor_random
# ours in random
ours_2024_random = [1000, 1000, 12, 1000, 18, 1000, 1000, 1000, 12, 1000]
ours_42_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ours_40226_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ours_40026_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

# pure ppo in random
ppo_42_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ppo_40226_random = [1000, 44, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ppo_2024_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

# rarl in random
rarl_42_random = [1000, 1000, 1000, 1000, 1000, 12, 1000, 1000, 1000, 72]
rarl_2024_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
rarl_422_random = [22, 46, 15, 125, 12, 11, 104, 145, 1000, 244] 
rarl_20242_random = [1000, 1000, 13, 15, 9, 1000, 7, 1000, 21, 14] 

# rap in random
rap_42_random = [1000, 1000, 1000, 10, 1000, 1000, 1000, 1000, 1000, 1000] 
rap_2024_random = [1000, 1000, 17, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
rap_422_random = [45, 16, 1000, 1000, 1000, 13, 1000, 12, 25, 1000] 

# # quadrotor_wind
# # ours in wind: 1 = hj_distbs = (0.00424, 0.0, 0.0); 2 = hj_distbs = (0.0, 0.00424, 0.0), 3:hj_distbs = (0.003, 0.003, 0.0)
# ours_42_wind1 = [1000, 1000, 1000, 1000, 4, 17, 1000, 11, 1000, 1000] 
# ours_42_wind2 = [1000, 1000, 1000, 17, 113, 1000, 1000, 48, 1000, 1000]
# ours_40226_wind1 = [10, 11, 1000, 1000, 1000, 1000, 7, 1000, 1000, 11] 
# ours_40226_wind2 = [7, 69, 1000, 1000, 1000, 6, 5, 1000, 85, 1000]
# ours_40026_wind1 = [46, 29, 16, 1000, 1000, 1000, 12, 1000, 1000, 9] 

ours_42_wind3 = [1000, 1000, 1000, 1000, 6, 1000, 1000, 10, 1000, 1000] 
# ours_40226_wind3 = [8, 66, 1000, 1000, 1000, 17, 10, 1000, 1000, 5]
ours_40026_wind3 =  [90, 16, 49, 1000, 1000, 1000, 1000, 1000, 1000, 83]
ours_45_wind3 = [1000, 1000, 1000, 1000, 5, 4, 1000, 11, 36, 1000]

# pure ppo in wind
ppo_42_wind3 = [1000, 33, 7, 1000, 1000, 1000, 1000, 5, 1000, 1000]
ppo_2024_wind3 = [1000, 1000, 1000, 1000, 13, 1000, 1000, 1000, 53, 1000] 
ppo_40226_wind3 = [5, 1000, 14, 6, 1000, 6, 22, 1000, 1000, 4]

# rap in wind
rap_42_wind3 = [68, 1000, 1000, 45, 11, 1000, 1000, 1000, 1000, 1000] 
rap_2024_wind3 = [1000, 1000, 22, 1000, 7,  1000, 33, 1000, 1000, 1000]
rap_422_wind3 =  [27, 44, 7, 23, 14, 585, 474, 5, 6, 250] 

# rarl in wind
rarl_42_wind3 = [7, 16, 1000, 6, 9, 27, 15, 1000, 1000, 12] 
rarl_2024_wind3 = [44, 30, 1000, 28, 6, 37, 1000, 29, 15, 38]
# rarl_422_wind3 = [12, 120, 4, 19, 422, 61, 51, 9, 11, 27]
rarl_20242_wind3 = [1000, 1000, 8, 1000, 8, 1000, 9, 1000, 6, 1000] 







if __name__ == '__main__':

    # Ours
    # ours_randomhj = [ours_42_randomhj, ours_2024_randomhj, ours_40226_randomhj]
    # ours_random = [ours_42_random, ours_2024_random, ours_40226_random]
    # print(f"========== Ours in the env quadrotor_randomhj:")
    # calculate_mean_std(ours_randomhj)
    # print(f"========== Ours in the env quadrotor_random:")
    # calculate_mean_std(ours_random)
    # print(f"========== Ours in the env quadrotor_wind3:")
    # calculate_mean_std([ours_42_wind3, ours_40026_wind3, ours_45_wind3])

    # Pure PPO
    # ppo_randomhj = [ppo_42_randomhj, ppo_2024_randomhj, ppo_40226_randomhj]
    # ppo_random = [ppo_42_random, ppo_2024_random, ppo_40226_random]
    # ppo_wind3 = [ppo_42_wind3, ppo_2024_wind3, ppo_40226_wind3]
    # print(f"========== Pure PPO in the env quadrotor_randomhj:")
    # calculate_mean_std(ppo_randomhj)
    # print(f"========== Pure PPO in the env quadrotor_random:")
    # calculate_mean_std(ppo_random)
    # print(f"========== Pure PPO in the env quadrotor_wind3:")
    # calculate_mean_std(ppo_wind3)
    
    # RARL
    rarl_randomhj = [rarl_42_randomhj, rarl_2024_randomhj, rarl_20242_randomhj]
    rarl_random = [rarl_42_random, rarl_2024_random, rarl_20242_random]
    rarl_wind3 = [rarl_42_wind3, rarl_2024_wind3, rarl_20242_random]
    print(f"========== RARL in the env quadrotor_randomhj:")
    calculate_mean_std(rarl_randomhj)
    print(f"========== RARL in the env quadrotor_random:")
    calculate_mean_std(rarl_random)
    print(f"========== RARL in the env quadrotor_wind3:")
    calculate_mean_std(rarl_wind3)
    
    # RAP
    rap_randomhj = [rap_42_randomhj, rap_2024_randomhj]
    rap_random = [rap_42_random, rap_2024_random]
    rap_wind3 = [rap_42_wind3, rap_2024_wind3]
    print(f"========== RAP in the env quadrotor_randomhj:")
    calculate_mean_std(rap_randomhj)
    print(f"========== RAP in the env quadrotor_random:")
    calculate_mean_std(rap_random)
    print(f"========== RAP in the env quadrotor_wind3:")
    calculate_mean_std(rap_wind3)
    
    
