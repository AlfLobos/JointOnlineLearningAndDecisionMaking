import torch
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    """
    Method to Create a Dataset in Pythorch
    """
    def __init__(self, X_Mat, Y_Vec):
        self.X_Mat = X_Mat 
        self.Y_Vec = Y_Vec

    def __len__(self):
        return self.X_Mat.size()[0]

    def __getitem__(self, idx):
        return [self.X_Mat[idx], self.Y_Vec[idx]]