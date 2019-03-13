
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
import torch

#Run this code to get the values of mean, standard deviation, standard deviation with delta degrees of freedom = 1.

my_transform = transforms.Compose([

    transforms.ToTensor()
])
# Define path to the data directory
root_dir = os.getcwd()
data_dir = os.path.join(os.path.join(root_dir, 'data'), 'chest_xray')

trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform= my_transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

data_mean = [] # Mean of the dataset
data_std0 = [] # std of dataset

for i, data in enumerate(dataloader, 0):
    # shape (batch_size, 3, height, width)
    numpy_image = data['image'].numpy()

    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))

    data_mean.append(batch_mean)
    data_std0.append(batch_std0)


# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
data_mean = np.array(data_mean).mean(axis=0)
data_std0 = np.array(data_std0).mean(axis=0)

print(data_mean, data_std0)

def get_mean_std(data):
    data = data.astype(np.float32) / 255.
    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:, i, :, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    return means, stdevs

#example with cifar10!
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()).train_data
print(get_mean_std(train_data))






