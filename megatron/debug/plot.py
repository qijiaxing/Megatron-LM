import numpy as np
import matplotlib.pyplot as plt
import torch

def calculate_percentiles(activations):
    min_vals = np.min(activations, axis=0)
    max_vals = np.max(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)
    return min_vals, max_vals, p1, p99, p25, p75

def plot_activation_distribution(tensor,
    save_image_path, save_image_name="Tensor Value Distribution"):
    """
    Plot the distribution of activations from a PyTorch tensor and save the plot as an image.

    Parameters:
    - tensor_path (str): Path to the PyTorch tensor file.
    - save_image_path (str): Path to save the generated image.
    - save_image_name (str): Title of the plot.
    """

    # Calculate percentiles and mean/std for the activations
    minv, maxv, p1, p99, p25, p75 = calculate_percentiles(tensor)

    # Create a figure and axis
    fig, axs = plt.subplots(1, 1, figsize=(7, 6))

    # Generate the hidden dimension index
    hidden_dimension_index = np.arange(tensor.shape[1])

    # Plot the distribution of activations
    axs.fill_between(hidden_dimension_index, minv, maxv, color='red', alpha=0.2, label='Min/Max')
    axs.fill_between(hidden_dimension_index, p1, p99, color='blue', alpha=0.3, label='1/99 Percentile')
    axs.fill_between(hidden_dimension_index, p25, p75, color='orange', alpha=0.5, label='25/75 Percentile')

    axs.set_title(f"{save_image_name}")
    axs.set_xlabel("Hidden dimension index")
    axs.set_ylabel("Value")

    # Add legend
    axs.legend(loc='upper right')

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Save figure to file: {save_image_path}")
    plt.savefig(save_image_path)


if __name__ == "__main__":
  filename = 'gpt3-8B-megatron-bf16-all-linear-layers/layer6.mlp.fc2/050000-of-262k/gpt3-8B-megatron-bf16-all-linear-layers.layer6.mlp.fc2.fc_save_x.050000-of-262k.pt'
  fig_filename = filename + ".dist.png"

  tensor = torch.load(filename).float()
  print(f"Load tensor from file: {filename}")
  print(f"Tensor shape: {tensor.shape}")
  h_dim = tensor.shape[-1]
  tensor = tensor.view(-1, h_dim).numpy()   # to 2D

  plot_activation_distribution(tensor, fig_filename)
