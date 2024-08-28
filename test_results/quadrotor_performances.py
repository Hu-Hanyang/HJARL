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
# rarl_42_randomhj 

# rap in randomhj

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

# rap in random
rap_42_random = [1000, 1000, 1000, 10, 1000, 1000, 1000, 1000, 1000, 1000] 

## quadrotor_wind
# ours in wind: 1 = hj_distbs = (0.00424, 0.0, 0.0); 2 = hj_distbs = (0.0, 0.00424, 0.0)
ours_42_wind1 = [1000, 1000, 1000, 1000, 4, 17, 1000, 11, 1000, 1000] 
ours_42_wind2 = [1000, 1000, 1000, 17, 113, 1000, 1000, 48, 1000, 1000]
ours_40226_wind1 = [10, 11, 1000, 1000, 1000, 1000, 7, 1000, 1000, 11] 
ours_40226_wind2 = [7, 69, 1000, 1000, 1000, 6, 5, 1000, 85, 1000]
# ours_40026_wind1 = [46, 29, 16, 1000, 1000, 1000, 12, 1000, 1000, 9] 







if __name__ == '__main__':

    # Ours
    # ours_randomhj = [ours_42_randomhj, ours_2024_randomhj, ours_40226_randomhj]
    # ours_random = [ours_42_random, ours_2024_random, ours_40226_random]
    # print(f"========== Ours in the env quadrotor_randomhj:")
    # calculate_mean_std(ours_randomhj)
    # print(f"========== Ours in the env quadrotor_random:")
    # calculate_mean_std(ours_random)

    # Pure PPO
    ppo_randomhj = [ppo_42_randomhj, ppo_2024_randomhj, ppo_40226_randomhj]
    ppo_random = [ppo_42_random, ppo_2024_random, ppo_40226_random]
    print(f"========== Pure PPO in the env quadrotor_randomhj:")
    calculate_mean_std(ppo_randomhj)
    print(f"========== Pure PPO in the env quadrotor_random:")
    calculate_mean_std(ppo_random)
