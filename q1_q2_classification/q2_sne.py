
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from voc_dataset import *
import random 
import torch 
import torchvision.models as models

from utils import *
from train_q2 import *
from sklearn.preprocessing import LabelEncoder



if __name__ =="__main__": 

    np.random.seed(16824)

    resnetCkpt = './checkpoint-model-epoch50.pth'

    device = torch.device("cuda")

    resnet = torch.load(resnetCkpt)

    # print(type(resnet))

    # print("before\n",resnet.state_dict().keys())   
    # print(list(resnet.children())[:-1])
    # Remove the fc layer to get outputs before fc layers
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    # print(type(resnet))
    # print("after\n", resnet.state_dict().keys())


    voc_loader = get_data_loader('voc', train=False, batch_size=16, split='test', inp_size=224)

    test_images = []
    test_labels = []

 
    for img, label, _ in voc_loader:
        # Append the entire batch
        test_images.append(img)
        test_labels.append(label)

    # Concatenate the lists to tensors
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    selected_indices = np.random.choice(len(test_images), size=1000, replace=False)

    # Select the subset of images and labels
    selected_images = test_images[selected_indices].to(device)
    selected_labels = test_labels[selected_indices]

    # Forward pass to get the features
    selected_features = []
    resnet.eval()
    with torch.no_grad():
        for img in selected_images:
            output = resnet(img.unsqueeze(0))
            selected_features.append(output.view(1, -1).cpu().numpy())

    selected_features = np.concatenate(selected_features, axis=0)

    # Apply t-SNE on numpy array
    embed_features = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=10000).fit_transform(selected_features)

    colors_rgb = np.array([(230, 25, 75),
                          (60, 180, 75),
                          (255, 225, 25),
                          (0, 130, 200),
                          (245, 130, 48),
                          (145, 30, 180),
                          (70, 240, 240),
                          (240, 50, 230),
                          (210, 245, 60),
                          (250, 190, 212),
                          (0, 128, 128),
                          (220, 190, 255),
                          (170, 110, 40),
                          (255, 250, 200),
                          (128, 0, 0),
                          (170, 255, 195),
                          (128, 128, 0),
                          (255, 215, 180),
                          (0, 0, 128),
                          (128, 128, 128)])
    
    
    colors_rgb = colors_rgb / 255.

    feature_colors = np.matmul(selected_labels.cpu().detach().numpy(), colors_rgb) / np.sum(selected_labels.cpu().detach().numpy(), axis=1).reshape(-1, 1)
    
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    custom_lines = []

    for i in range(20):
        custom_lines.append(plt.Line2D([0],
                                      [0],
                                      marker='o',
                                      color=colors_rgb[i],
                                      label=CLASS_NAMES[i],
                                      markersize=10))

    # Plot the tsne features with colors
    plt.figure(figsize = (12,12))
    plt.scatter(embed_features[:, 0], embed_features[:, 1], c=feature_colors)
    plt.legend(handles=custom_lines)
    plt.title("ResNet Visualizations")
    plt.show()
        



