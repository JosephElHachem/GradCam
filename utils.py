import numpy as np
import torch
import cv2
from torchvision import models, datasets, transforms

# Define preprocessing function of the input images
def preprocess_image(dir_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(dir_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # resize the image to 224x224
            transforms.ToTensor(), # convert numpy.array to tensor
            normalize])) #normalize the tensor

    return (dataset)

def denormalize(img):
    mean=torch.Tensor([0.485, 0.456, 0.406])
    std=torch.Tensor([0.229, 0.224, 0.225])
    img[:, :, 0] = img[:, :, 0] * std[0] + mean[0]
    img[:, :, 1] = img[:,: , 1] * std[1] + mean[1]
    img[:, :, 2] = img[:,: , 2] * std[2] + mean[2]
    return img

def heat_image(image, heat_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * heat_map), cv2.COLORMAP_JET)
    heated = cv2.addWeighted(heatmap, 0.3, np.uint8(255*image), 0.7, 0)
    return heated


# not used
def rgb2gray(img):
    gray_img = 0.299* img[:, :, 0] + 0.587 *img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray_img

# not used
def mask(img,i):
    img2 = torch.zeros(img.size())
    img2[:, :, i] = img[:, :, i]
    return img2

# not used
def add_heatmap_rgb(img,heat_map):
    new_image = torch.zeros((224,224,3))
    new_image[:,:,0] = (img[:,:,0] + heat_map)/2
    new_image[:,:,1] = (img[:,:,1] + heat_map)/2
    new_image[:,:,2] = (img[:,:,2] + heat_map)/2
    return new_image

# not used
def add_heatmap(img,heat_map):
    new_image = torch.zeros((224,224,3))
    new_image[:,:,0] = img
    new_image[:,:,1] = (heat_map + img)/2
    new_image[:,:,2] = heat_map
    return new_image
