import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.head(x)

def prepare_grid_for_ai(grid):    
    new_grid = np.zeros_like(grid, dtype=float)
    
    mask = grid > 0
    new_grid[mask] = np.log2(grid[mask])
    
    flattened = new_grid.flatten()
    
    return torch.tensor(flattened, dtype=torch.float32).unsqueeze(0)