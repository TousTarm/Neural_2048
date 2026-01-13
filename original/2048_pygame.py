import pygame
import sys
import random
import numpy as np

WIDTH = 400
HEIGHT = 400
BLOCK_SIZE = 100
BG_COLOR = (187, 173, 160)
FONT_COLOR = (119, 110, 101)

COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

grid = np.zeros((4, 4), dtype=int)

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
    
    return grid

def draw_window(screen, font):
    global grid
    screen.fill(BG_COLOR)
    
    for r in range(4):
        for c in range(4):
            value = grid[r, c] # NumPy přístup k souřadnicím
            
            # Vykreslení čtverečku
            color = COLORS.get(value, COLORS[2048])
            rect = pygame.Rect(c * BLOCK_SIZE, r * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BG_COLOR, rect, 5) # Tlustší rámeček vypadá líp
            
            # Vykreslení čísla
            if value != 0:
                # Menší font pro velká čísla (1024, 2048)
                current_font = font
                if value > 512:
                    current_font = pygame.font.SysFont("arial", 30, bold=True)
                
                text_color = (119, 110, 101) if value < 8 else (249, 246, 242)
                text_surface = current_font.render(str(value), True, text_color)
                text_rect = text_surface.get_rect(center=(c * BLOCK_SIZE + BLOCK_SIZE/2, r * BLOCK_SIZE + BLOCK_SIZE/2))
                screen.blit(text_surface, text_rect)
    
    pygame.display.update()

def main():
    # Inicializace
    init_grid()
    pygame.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural 2048 (NumPy Edition)")
    font = pygame.font.SysFont("arial", 45, bold=True)
    clock = pygame.time.Clock()
    
    running = True
    
    # Hlavní smyčka
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game_step("a")
                elif event.key == pygame.K_RIGHT:
                    game_step("d")
                elif event.key == pygame.K_UP:
                    game_step("w")
                elif event.key == pygame.K_DOWN:
                    game_step("s")
                elif event.key == pygame.K_r:
                    init_grid()
        
        draw_window(screen, font)
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()