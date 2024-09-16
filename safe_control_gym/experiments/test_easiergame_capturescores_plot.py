import numpy as np
import matplotlib.pyplot as plt
from odp.Grid import Grid
from safe_control_gym.utils.plotting import po2slice1vs1
from matplotlib.colors import LinearSegmentedColormap
    

fixed_defender_position = np.array([[0.5, 0.0]])  # np.array([[-0.5, -0.5]]), np.array([[0.0, 0.0]])
# fixed_defender_position = np.array([[-0.5, -0.5]])  # np.array([[-0.5, -0.5]]), np.array([[0.0, 0.0]])

x_range = np.arange(-0.95, 1.0, 0.05)  # from -0.95 to 0.95
y_range = np.arange(-0.95, 1.0, 0.05)

# ours scores
loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[0.5 0.].npy')
# loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[-0.5 -0.5].npy')

# rarl scores
# loaded_scores = np.load('training_results/rarl_game/rarl/seed_42/score_matrix_[0.5, 0.0].npy')
# loaded_scores = np.load('training_results/rarl_game/rarl/seed_42/score_matrix_[-0.5, -0.5].npy')

# rap scores
# loaded_scores = np.load('training_results/rarl_game/rap/seed_2024/score_matrix_[0.5, 0.0].npy')
# loaded_scores = np.load('training_results/rarl_game/rap/seed_2024/score_matrix_[-0.5, -0.5].npy')

# Process
loaded_scores = loaded_scores.T
# print(loaded_scores)
# Create coordinates for the scatter plot
x, y = np.meshgrid(np.arange(loaded_scores.shape[1]), np.arange(loaded_scores.shape[0]))
x = x.flatten()
y = y.flatten()
values = loaded_scores.flatten()

# Ensure that x_range and y_range cover from -1.0 to 1.0 for plotting purposes
extended_x_range = np.linspace(-1.0, 1.0, len(x_range))
extended_y_range = np.linspace(-1.0, 1.0, len(y_range))

# Plot the HJ value function
value1vs1 = np.load(('safe_control_gym/envs/gym_game/values/1vs1Defender_easier.npy'))
grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))

initial_attacker = np.array([[-0.5, 0.0]])
a1x_slice, a1y_slice, d1x_slice, d1y_slice = po2slice1vs1(initial_attacker[0], fixed_defender_position[0], value1vs1.shape[0])
value_function1vs1 = value1vs1[:, :, d1x_slice, d1y_slice].squeeze()
value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)
# print(f"The shape of the value_function1vs1 is {value_function1vs1.shape}")
dims_plot = [0, 1]
dim1, dim2 = dims_plot[0], dims_plot[1]
x_hj = np.linspace(-1, 1, value_function1vs1.shape[dim1])
y_hj = np.linspace(-1, 1, value_function1vs1.shape[dim2])

# Define a custom colormap
colors = [(0.28, 0.34, 0.77), (0.8, 0.8, 1), (0.94, 0.72, 0.6)]  
n_bins = 100  # Number of bins
cmap_name = 'custom_contour'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# # Create the 2D plot
plt.figure(figsize=(8, 8))
plt.imshow(loaded_scores, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap=cm)
# plt.imshow(loaded_scores, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='Pastel2')  # cmap='viridis', Pastel1,Pastel2

# Add color bar to indicate the score values
# plt.colorbar(label='Scores')
plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='magenta', marker='*', s=300, label='Fixed Defender')
contour = plt.contour(x_hj, y_hj, value_function1vs1, levels=0, colors='#4B0082', linewidths=3.0, linestyles='dashed')  # colors='magenta'


# # Hanyang: try scatters 
# plt.scatter(x[values == -1.0], y[values == -1.0], color='blue', edgecolor='k', label='-1.0')

# Add labels and titles
fontdict1 = {'fontsize': 30, 'fontweight': 'bold'}
fontdict2 = {'fontsize': 40}  # , 'fontweight': 'bold'

plt.xlabel('X', fontdict=fontdict1)
plt.ylabel('Y', fontdict=fontdict1)
plt.title("HJARL (ours)", fontdict=fontdict2)

# plt.title("RARL [8]", fontdict=fontdict2)
# plt.title("RAP [20]", fontdict=fontdict2)

# Set the title of the plot
# plt.title('2D Plot of Scores')
plt.savefig(f'training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/hjarl_score_matrix_{fixed_defender_position[0]}.png')
# plt.savefig(f'training_results/rarl_game/rarl/seed_42/rarl_score_matrix_{fixed_defender_position[0]}.png')
# plt.savefig(f'training_results/rarl_game/rap/seed_2024/rap_score_matrix_{fixed_defender_position[0]}.png')
# Show the plot
plt.show()
