import matplotlib.pyplot as plt

def plot_last_epoch_losses(topK_values, last_epoch_losses, save_path=None):
    plt.figure(figsize=(10, 6))

    for i, topK_value in enumerate(topK_values):
        plt.plot(topK_value, last_epoch_losses[i], marker='o', label=f'TopK Value {topK_value}')

    plt.title('BCE Loss vs TopK Values (Last Epoch)')
    plt.xlabel('TopK Values')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_last_epoch_losses(topK_values, last_epoch_losses, save_path=None):
    plt.figure(figsize=(10, 6))

    for i, topK_value in enumerate(topK_values):
        plt.plot(topK_value, 1 - last_epoch_losses[i], marker='o', label=f'TopK Value {topK_value}')

    plt.title('1 - BCE Loss vs TopK Values (Last Epoch)')
    plt.xlabel('TopK Values')
    plt.ylabel('1 - BCE Loss')
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        

def plot_losses(topK_values, losses, save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Transpose the list of losses for correct plotting
    losses_transposed = list(map(list, zip(*losses)))

    for i, loss_values in enumerate(losses_transposed):
        plt.plot(topK_values, loss_values, marker='o', label=f'Epoch {i + 1}')

    plt.title('BCE Loss vs TopK Values')
    plt.xlabel('TopK Values')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_feature_maps_(feature_maps, title, save_path=None):
    # Take a slice of the feature maps (e.g., first channel)
    sliced_feature_maps = feature_maps[:, :1, :, :]  

    # Create a grid of feature maps for visualization
    grid = make_grid(sliced_feature_maps, normalize=True, scale_each=True, nrow=8)
    
    # Convert to NumPy array and transpose to (H, W, C) for matplotlib
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Display 
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_feature_maps(feature_maps, title, save_path=None):
    # Normalize each channel to [0, 1]
    feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())
    feature_maps = feature_maps.view(feature_maps.shape[2], feature_maps.shape[3], feature_maps.shape[1])
    feature_maps_np = feature_maps.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(feature_maps_np)
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()