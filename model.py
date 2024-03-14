import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torchvision import transforms
from scipy.ndimage import center_of_mass, shift


class MLP(nn.Module):
    def __init__(self, layers=1, width=64, inference=False, device="cpu"):
        super(MLP, self).__init__()

        dims = [784] + [width] * layers + [10]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers + 1)])
        self.inference = inference

        self.to(device)

    def forward(self, x):
        layer_activations = []
        if self.inference:
            layer_activations.append(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                x = F.relu(x)

            if self.inference:
                layer_activations.append(x)

        if not self.inference:
            output = F.log_softmax(x, dim=1)
            return output

        output = F.softmax(x, dim=1)
        return output, layer_activations


def center_image(img):
    if img is None or np.mean(img) == 0.0:
        return img
    com_x, com_y = center_of_mass(img)
    shift_x = img.shape[0] // 2 - com_x
    shift_y = img.shape[1] // 2 - com_y
    centered_img = shift(img, (shift_x, shift_y), mode='constant', cval=0)
    return centered_img


transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,)),
  transforms.Lambda(torch.flatten)
])


center_transform = transforms.Compose([
    transforms.Lambda(center_image),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(torch.flatten)
])
