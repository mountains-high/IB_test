import torch
import torch.nn as nn


class Feature_Decoder_full(nn.Module):
    def __init__(self):
        super(Feature_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(512, 256, 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(256, 128, 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(128, 64, 1, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        self.conv5 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        self.conv6 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
        self.conv_31 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_34 = nn.Conv2d(3, 3, 3, stride=1, padding=1)

    def forward(self, x, f1, f2, f3, f4, f5):
        out = self.conv1(self.upsample(f5))
        out = self.conv_31(out + f4)
        
        out = self.conv2(self.upsample(out))
        out = self.conv_32(out + f3)
        
        out = self.conv3(self.upsample(out))
        out = self.conv_33(out + f2)
        
        out_ = self.conv4(out)
        out = (x + out_)
        out = self.conv6(out)
        out = torch.nn.functional.tanh(out)
        
        return out, out_


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




