import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from train_q2 import ResNet
from voc_dataset import VOCDataset

# Load your test dataset with images and ground truth class labels
# Replace this with code to load your dataset
test_images = []  # List of PIL images
test_labels = []  # List of ground truth class labels

# Initialize a pre-trained ResNet model and feature extractor
# Replace this with code to initialize your model
model = ResNet(num_classes=len(VOCDataset.CLASS_NAMES))  # Change num_classes based on your model

# Initialize a TSNE object for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)

# Extract features from the images
features = []
for img in test_images:
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)  # Add batch dimension
    features.append(model(img).detach().numpy())

features = np.vstack(features)

# Perform t-SNE dimensionality reduction
reduced_features = tsne.fit_transform(features)

# Calculate the mean color for each image's classes
# Replace this with code to get class labels for each image
image_class_labels = []  # List of class labels for each image

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
