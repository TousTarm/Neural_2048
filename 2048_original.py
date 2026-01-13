import pygame
import random
import sys

grid = [[0 for _ in range(4)] for _ in range(4)]

def print_grid():
    global grid
    print("")
    for line in grid:
        print(line)
    
def init_grid():
    global grid
    spots = random.sample(range(16), 2)
    for spot in spots:
        r = spot // 4
        c = spot % 4
        grid[r][c] = generate_cell_value() 
    return grid

def generate_cell_value():
    return 4 if random.randint(0, 9) == 0 else 2 

def game_step(direction):
    global grid
    legal = True
    if direction == "a":
        if check_if_legal():
            move_left()
        else:
            legal = False

    if direction == "d":
        reverse()
        if check_if_legal():
            move_left()
        else:
            legal = False
        reverse()

    if direction == "w":
        transpose()
        if check_if_legal():
            move_left()
        else:
            legal = False
        transpose()

    if direction == "s":
        transpose()
        reverse()
        if check_if_legal():
            move_left()
        else:
            legal = False
        reverse()
        transpose()

    generate_cell() if legal else 0
    print_grid()
    return grid
    
def transpose():
    global grid
    grid = [list(row) for row in zip(*grid)]

def reverse():
    global grid
    grid = [row[::-1] for row in grid]

def move_left():
    global grid
    new_grid = []
    for line in grid:
        new_line = line[:]
        while(0 in new_line):
            new_line.remove(0)
        
        merged_line = new_line
        for x in range(len(new_line)):
            if x+1 < len(new_line):
                if merged_line[x] == merged_line[x+1]:
                    merged_line[x] *= 2
                    del merged_line[x+1]
                    merged_line.append(0)
            

        while(len(new_line) < 4):
            new_line.append(0)
        new_grid.append(new_line)
    grid = new_grid

def generate_cell():
    global grid
    empty_cells = 0
    for line in grid:
        empty_cells += line.count(0)
    if empty_cells == 0:
        print("game over")
        return grid
    spot = random.randint(0, empty_cells - 1)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 0:
                if spot == 0:
                    grid[i][j] = generate_cell_value()
                    return grid
                spot -= 1
    return grid
    
def check_if_legal():
    global grid
    for line in grid:
        for x in range(len(line)):
            if x+1 < len(line):
                if line[x] == 0 and line[x+1] != 0:
                    return True
    
    
    for line in grid:
        for x in range(len(line)):
            if x+1 < len(line):
                if line[x] == line[x+1] and line[x] != 0:
                    return True
    print("ilegal")
    return False
    
        


init_grid()
print_grid()

while(True):
    direction = input("enter move\n")
    game_step(direction)