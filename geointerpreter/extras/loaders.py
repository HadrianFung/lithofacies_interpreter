import os
import pickle
import torch
import torch.nn as nn

class ModelLoaders():
    """
    Class to load the trained models
    """
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    def load_model(self, model):
        """
        Function to load the trained model
        
        Parameters
        ----------
        model : str
            Name of the model to be loaded
        
        Returns
        -------
        model : sklearn or PyTorch model
            Loaded model
        """

        if model == "RandomForest":
            return pickle.load(open(os.path.join(self.model_dir, 'perm_model_RF.pkl'), 'rb'))
        elif model == "SVR":
            return pickle.load(open(os.path.join(self.model_dir, 'perm_model_SVR.pkl'), 'rb'))
        elif model == "XGB":
            return pickle.load(open(os.path.join(self.model_dir, 'perm_model_XGB.pkl'), 'rb'))
        elif model == "ConcatCNN":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model = ConcatCNN()
            cnn_model.load_state_dict(torch.load(os.path.join(self.model_dir, 'best_model_23.pth'), map_location=device))
            cnn_model.to(device)
            cnn_model.eval()
            return cnn_model

class ConcatCNN(nn.Module):
    """
    Class to construct the model architecture
    """
    def __init__(self):

        super(ConcatCNN, self).__init__()
        
        # Convolutional pathways for image data
        self.cnn_path = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU()
        )

        # Fully connected pathway for numerical data from wireline logs
        self.data_path = nn.Sequential(
            nn.Linear(11, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,512),
            nn.LeakyReLU(),
        )

        # Merge and export in fully connected layer
        # Output layer as the number of facies to be classified
        self.merged_path = nn.Sequential(
            nn.Linear(120 + 512, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, image_input, data_input):
        image_out = self.cnn_path(image_input)
        data_out = self.data_path(data_input)
        merged = torch.cat((image_out, data_out), dim=1)
        out = self.merged_path(merged)
        return out
