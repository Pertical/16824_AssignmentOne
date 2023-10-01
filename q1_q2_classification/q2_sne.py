
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from voc_dataset import *
import random 
import torch 
import torchvision.models as models

from utils import *
from train_q2 import ResNet
from sklearn.preprocessing import LabelEncoder


# voc_data = VOCDataset(split='test', size = 224)

voc_loader = get_data_loader('voc', train=False, batch_size=1, split='test', inp_size=224)

test_images = []
test_labels = []


for i in range(1000):
    for img, label, _ in voc_loader:
        img = img.numpy()  # Convert to NumPy array
        img = torch.from_numpy(img).detach()  # Detach from gradients
        test_images.append(img)
        test_labels.append(label)

model = ResNet(len(VOCDataset.CLASS_NAMES)).to(use_cuda=True)
model.load_state_dict(torch.load('checkpoint-model-epoch50.pth'))
model.eval()

tsne = TSNE(n_components=2, random_state=0)

features = []
for img in test_images:
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation during inference
        feature_vector = model(img).numpy()  # Extract features using the model
    features.append(feature_vector)

features = np.vstack(features)
reduced_features  = tsne.fit_transform(features)

image_class_labels = test_labels  # Use the loaded labels


distinct_colors = plt.cm.get_cmap('tab20', len(VOCDataset.CLASS_NAMES))
mean_colors = []
label_encoder = LabelEncoder()

for labels in image_class_labels:
    class_indices = label_encoder.transform(labels)
    class_colors = [distinct_colors[i] for i in class_indices]
    mean_color = np.mean(class_colors, axis=0)
    mean_colors.append(mean_color)

# Plot the 2D t-SNE projection with color-coded points
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=mean_colors)

# Create a legend explaining the mapping from color to object class
class_names = label_encoder.classes_
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=class_name)
                  for class_name, color in zip(class_names, distinct_colors)]

plt.legend(handles=legend_handles, title="Object Class")

# Show the plot
plt.show()
