import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- OČI (Konvoluční vrstvy) ---
        # 1. vrstva: Vidí základní vztahy (např. dvě stejná čísla vedle sebe)
        # Vstup: 1 kanál (mřížka), Výstup: 64 filtrů, Okno: 2x2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        
        # 2. vrstva: Skládá složitější vzory z těch 64 filtrů
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        
        # --- MOZEK (Lineární vrstvy) ---
        # Spočítáme velikost vstupu pro lineární vrstvu.
        # Pro vstup 4x4, kernel 2, padding 1 nám po konvolucích vyjde rozměr 5x5.
        # 128 (filtrů) * 5 * 5 = 3200
        self.fc_input_dim = 128 * 6 * 6
        
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.head = nn.Linear(256, 4) # 4 směry

    def forward(self, x):
        # x přichází jako (Batch, 16) z paměti, musíme to vrátit na 2D (Batch, 1, 4, 4)
        x = x.view(-1, 1, 4, 4) 
        
        # Aplikace očí
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Zploštění: Převod z 3D kostky dat na 1D řadu pro rozhodování
        x = x.view(x.size(0), -1) 
        
        # Rozhodování
        x = F.relu(self.fc1(x))
        return self.head(x)

def prepare_grid_for_ai(grid):    
    new_grid = np.zeros_like(grid, dtype=float)
    
    # Logaritmování (aby čísla nebyla obrovská)
    mask = grid > 0
    new_grid[mask] = np.log2(grid[mask])
    
    # --- ZMĚNA PRO CNN ---
    # Už neděláme flatten! Chceme zachovat tvar 4x4.
    tensor = torch.tensor(new_grid, dtype=torch.float32)
    
    # Přidáme rozměry pro Batch a Kanál -> (1, 1, 4, 4)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor