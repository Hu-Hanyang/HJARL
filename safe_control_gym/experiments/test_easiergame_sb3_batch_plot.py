import numpy as np
import matplotlib.pyplot as plt


def plot_batch_scores(fixed_defender, scores, x_range, y_range, save_path=None):
    '''Plot the batch scores in 2D.'''
    # Ensure that x_range and y_range cover from -1.0 to 1.0 for plotting purposes
    extended_x_range = np.linspace(-1.0, 1.0, len(x_range))
    extended_y_range = np.linspace(-1.0, 1.0, len(y_range))

    # Create the 2D plot
    plt.figure(figsize=(10, 8))
    plt.imshow(scores, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='viridis')

    # Add color bar to indicate the score values
    plt.colorbar(label='Scores')
    plt.scatter(fixed_defender[0][0], fixed_defender[0][1], color='red', marker='*', label='Fixed Defender')

    # Set the x and y axis labels
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Set the title of the plot
    plt.title('2D Plot of Scores')

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    plt.show()
    


fixed_defender_position = np.array([[-0.5, -0.5]])
x_range = np.arange(-0.95, 1.0, 0.05)  # from -0.95 to 0.95
y_range = np.arange(-0.95, 1.0, 0.05)
# loaded_scores = np.load(f'training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_{fixed_defender_position[0]}.npy')
# loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[-0.5 0.].npy')
# loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[-0.5 0.5].npy')
loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[-0.5 -0.5].npy')
# loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[0.5 0.].npy')
# loaded_scores = np.load('training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_[0. 0.].npy')

loaded_scores = loaded_scores.T
# print(loaded_scores.shape)


# Ensure that x_range and y_range cover from -1.0 to 1.0 for plotting purposes
extended_x_range = np.linspace(-1.0, 1.0, len(x_range))
extended_y_range = np.linspace(-1.0, 1.0, len(y_range))

# Create the 2D plot
plt.figure(figsize=(10, 8))
plt.imshow(loaded_scores, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='viridis')

# Add color bar to indicate the score values
plt.colorbar(label='Scores')
plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='red', marker='*', label='Fixed Defender')

# Set the x and y axis labels
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Set the title of the plot
plt.title('2D Plot of Scores')
plt.savefig(f'training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_{fixed_defender_position[0]}.png')
# Show the plot
plt.show()
