import numpy as np
import random

def print_grid():
    global grid
    print("-" * 20)
    print(grid)
    print("-" * 20)

def init_grid():
    global grid
    grid = np.zeros((4, 4), dtype=int)
    spots = random.sample(range(16), 2)
    for spot in spots:
        r = spot // 4
        c = spot % 4
        grid[r, c] = generate_cell_value()
    return grid

def generate_cell_value():
    return 4 if random.randint(0, 9) == 0 else 2 

def transpose():
    global grid
    grid = grid.T

def reverse():
    global grid
    grid = grid[:, ::-1]

def move_left():
    global grid
    new_grid = np.zeros((4, 4), dtype=int)
    
    for i in range(4):
        row = grid[i]
        non_zeros = row[row != 0]
        for j in range(len(non_zeros) - 1):
            if non_zeros[j] == non_zeros[j+1] and non_zeros[j] != 0:
                non_zeros[j] *= 2
                non_zeros[j+1] = 0
        new_row = non_zeros[non_zeros != 0]
        new_grid[i, :len(new_row)] = new_row
    grid = new_grid

def check_if_legal():
    global grid
    zeros_check = (grid[:, :-1] == 0) & (grid[:, 1:] != 0)
    merge_check = (grid[:, :-1] == grid[:, 1:]) & (grid[:, :-1] != 0)
    if np.any(zeros_check) or np.any(merge_check):
        return True
    return False

def generate_cell():
    global grid
    empty_cells = np.argwhere(grid == 0)
    
    if len(empty_cells) == 0:
        print("GAME OVER")
        return grid
        
    idx = random.randint(0, len(empty_cells) - 1)
    r, c = empty_cells[idx]
    
    grid[r, c] = generate_cell_value()
    return grid

def game_step(direction):
    global grid
    legal = False

    if direction == "a":
        if check_if_legal():
            move_left()
            legal = True

    elif direction == "d":
        reverse()
        if check_if_legal():
            move_left()
            legal = True
        reverse()

    elif direction == "w":
        transpose()
        if check_if_legal():
            move_left()
            legal = True
        transpose()

    elif direction == "s":
        transpose()
        reverse()
        if check_if_legal():
            move_left()
            legal = True
        reverse()
        transpose()

    if legal:
        generate_cell()
    else:
        print("Neplatný tah!")
        
    print_grid()
    return grid

init_grid()
print_grid()
print("START (Ovládání: w, a, s, d + Enter)\n")

while True:
    try:
        direction = input("move: ").strip().lower()
        if direction in ['w', 'a', 's', 'd']:
            game_step(direction)
    except KeyboardInterrupt:
        print("\nKonec hry.")
        break