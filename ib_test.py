import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from resnet import ResNet34
from plots import plot_last_epoch_losses

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)


class Feature_Decoder(nn.Module):
    def __init__(self, topK):
        super(Feature_Decoder, self).__init__()
        self.topK = topK
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv_31 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_34 = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        
    def apply_sparsity(self, tensor):
        if self.topK == 0.0:
            # Set all elements to zero if topK is 0.0
            tensor = torch.zeros_like(tensor)
        elif self.topK == 1.0:
            # Keep all features if topK is 1.0
            pass
        else:
            total_elements = tensor.numel()
            topK_value = int(self.topK * total_elements)

            if topK_value > total_elements:
                topK_value = total_elements

            _, top_indices = torch.topk(tensor.view(-1), k=topK_value)

            # Creating a mask for the selected indices
            mask = torch.zeros_like(tensor.view(-1))
            mask[top_indices] = 1.0

            # Applying the mask to zero out values in the tensor
            tensor = tensor * mask.view(tensor.shape)
        
        return tensor


    def forward(self, x, f1, f2, f3, f4, f5):
        # Apply sparsity to f5
        f5 = self.apply_sparsity(f5)

        # Upsample and conv1
        out = self.conv1(self.upsample(f5))
        out = self.conv_31(out + f4)

        # Upsample and conv2
        out = self.conv2(self.upsample(out))
        out = self.conv_32(out + f3)

        # Upsample and conv3
        out = self.conv3(self.upsample(out))
        out = self.conv_33(out + f2)

        # Final convolutions
        out_ = self.conv4(out)
        out = (x + out_)
        out = self.conv6(out)
        out = torch.nn.functional.tanh(out)

        return out, out_



def bce(inputs_org, inputs_bar):
    inputs_org_flat = inputs_org.view(inputs_org.size(0), -1)
    inputs_bar_flat = inputs_bar.view(inputs_bar.size(0), -1)
    bce_loss = nn.BCEWithLogitsLoss()(inputs_bar_flat, inputs_org_flat)

    return bce_loss


def get_images_initial(net, inputs_org, epochs=1, d_lr=0.0001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # list of topK 
    topK_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    all_losses = []

    inputs_org = inputs_org.clone().detach().to(device)

    for topK in topK_values:
        feature = Feature_Decoder(topK=topK).to(device)
        optimizer_f = torch.optim.Adam(feature.parameters(), lr=d_lr)
        epoch_losses = []

        for epoch in tqdm(range(epochs), leave=False, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            with torch.no_grad():
                _, f5, f4, f3, f2, f1 = net(inputs_org)
                
            # Visualize the original image
            original_image_grid = make_grid(inputs_org, nrow=1, normalize=True)
            original_image_np = original_image_grid.cpu().numpy().transpose(1, 2, 0)
            plt.figure(figsize=(6, 6))
            plt.imshow(original_image_np)
            plt.title('Original Image')
            plt.savefig('Original Image')
                
            # Visualize the original feature maps before applying sparsity
            #visualize_feature_maps(inputs_org, 'Original Feature Maps', save_path='original_feature_maps.png')

            # Pseudo input from the teacher's decoding layer
            inputs_bar, addition = feature(inputs_org, f1, f2, f3, f4, f5)

            # Visualize the feature maps after applying sparsity
            #visualize_feature_maps(inputs_bar, f'Feature Maps After Sparsity (TopK={topK})', save_path=f'feature_maps_after_sparsity_topK_{topK}.png')


            # Compute and store BCE loss
            loss_bce = bce(inputs_org, inputs_bar)
            epoch_losses.append(loss_bce.item())

            optimizer_f.zero_grad()
            loss_bce.backward()
            optimizer_f.step()

        all_losses.append(epoch_losses)

    # Plotting losses with topK values on the y-axis
    plot_losses(topK_values, all_losses, save_path='bce_losses.png')


def get_images(net, inputs_org, epochs=1, d_lr=0.0001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    topK_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    last_epoch_losses = []

    inputs_org = inputs_org.clone().detach().to(device)

    for topK in topK_values:
        feature = Feature_Decoder(topK=topK).to(device)
        optimizer_f = torch.optim.Adam(feature.parameters(), lr=d_lr)
        epoch_losses = []

        for epoch in tqdm(range(epochs), leave=False, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            with torch.no_grad():
                _, f5, f4, f3, f2, f1 = net(inputs_org)

            inputs_bar, addition = feature(inputs_org, f1, f2, f3, f4, f5)

            loss_bce = bce(inputs_org, inputs_bar)
            epoch_losses.append(loss_bce.item())

            optimizer_f.zero_grad()
            loss_bce.backward()
            optimizer_f.step()

        last_epoch_losses.append(epoch_losses[-1]) 

    # Plotting losses with topK values on the y-axis
    plot_last_epoch_losses(topK_values, last_epoch_losses, save_path='bce_losses_last_epoch.png')
   
if __name__ == "__main__":
    net_teacher = ResNet34()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_teacher = net_teacher.to(device)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='../data/CIFAR10', download = True, train=False, transform=transform_test)

    # Filter out only images of dogs (class label 5 in CIFAR-10)
    dog_indices = [i for i, (_, label) in enumerate(testset) if label == 5]
    dog_loader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(dog_indices), shuffle=False)

    # Set seed for data loader
    torch.manual_seed(seed)
    dog_loader = torch.utils.data.DataLoader(testset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(dog_indices), shuffle=False)

    # to get the original input image
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dog_loader):
            inputs_org, targets = inputs.to(device), targets.to(device)

    get_images(net=net_teacher,
               inputs_org=inputs_org,
               epochs=10,
               d_lr=0.0001)
