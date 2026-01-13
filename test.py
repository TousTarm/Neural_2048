import numpy as np
from ai import NeuralNetwork, prepare_grid_for_ai

model = NeuralNetwork()

grid = np.array([
    [0, 0, 0, 2],
    [0, 4, 1024, 0],
    [2, 2, 0, 0],
    [0, 0, 8, 16]
])

vstup = prepare_grid_for_ai(grid)

vystup = model(vstup)

print("Vstup pro AI (Log2):")
print(vstup)
print("\nVýstup AI (Hodnocení tahů [Left, Up, Right, Down]):")
print(vystup)