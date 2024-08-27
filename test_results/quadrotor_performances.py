import numpy as np


def calculate_mean_std(list):
    # Combine the lists into a single array
    combined_data = np.array(list[0] + list[1] + list[2])
    
    # Calculate mean and standard deviation
    mean = np.mean(combined_data)
    std_dev = np.std(combined_data)
    print(f"========== mean = {mean}, std_dev = {std_dev} ========== \n")
    
    # return mean, std_dev


### Data loading: method + method random seed + test env, eg, ours_42_randomhj
# ours in randomhj
ours_42_randomhj = [1000, 1000, 1000, 1000, 12, 2, 1000, 1000, 4, 1000] 
ours_2024_randomhj = [4, 1000, 15, 1000, 1000, 5, 1000, 1000, 1000, 2] 
ours_40226_randomhj = [1000, 1000, 1000, 1000, 2, 24, 1000, 1000, 1000, 1000] 
# pure ppo in randomhj
ppo_42_randomhj = [3, 1000, 29, 49, 15, 4, 1000, 119, 5, 89] 
# rarl in randomhj
# rarl_42_randomhj 

# rap in randomhj

# ours in random
ours_2024_random = [1000, 1000, 12, 1000, 18, 1000, 1000, 1000, 12, 1000]
ours_42_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ours_40226_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
ours_40026_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

# pure ppo in random
ppo_42_random = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]


if __name__ == '__main__':
    
    ours = [ours_42_randomhj, ours_2024_randomhj, ours_40226_randomhj]

    print(f"========== Ours in the env quadrotor_randomhj:")
    calculate_mean_std(ours)
