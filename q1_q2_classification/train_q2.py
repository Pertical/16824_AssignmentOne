import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        
        # self.fc = nn.Linear(1000, num_classes)

        # self.resnet.fc = nn.Linear(1000, num_classes)


        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return unnormalized log-probabilities here
        ##################################################################
        x = self.resnet(x)
        x = self.fc(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################




# def plot_tsne(model, checkpoint, loader, device):
# ##################################################################
#     from sklearn.manifold import TSNE

#     n_classes = 20 

#     dataset = VOCDataset('test', 224)
#     total_test_images = len(dataset)

#     print("total_test_images", total_test_images)
#     indices = np.randint(0, total_test_images, (1000,))
#     print("indices", indices)

#     labels = []
#     for i in indices:
#         labels.append((dataset[i][1]).reshape(1,n_classes))
    
#     labels = np.vstack(labels)
#     print("labels", labels.shape)

#     model = ResNet(len(VOCDataset.CLASS_NAMES)).to(device)
#     model.load_state_dict(torch.load('checkpoint-model-epoch50.pth'))

#     for name, module in model.named_modules():
#         if name == 'resnet':
#             print("module", module)
#             module.register_forward_hook(get_activation('resnet'))


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    args = ARGS(
        epochs=100,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=0.00001,
        batch_size=16,
        step_size=10,
        gamma=0.05, 
        #added to save the finetuned model. 
        save_at_end = True, 
        save_freq = 10 
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)
    

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

# import torch

# # Check if CUDA is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU")
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# # Now, you can use 'device' for your model and data transfers
