import numpy as np

class GameEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()

        return self.grid

    def add_random_tile(self):
        empty_cells = np.argwhere(self.grid == 0)
        if len(empty_cells) == 0: return
        idx = np.random.randint(len(empty_cells))
        r, c = empty_cells[idx]
        self.grid[r, c] = 2 if np.random.random() < 0.9 else 4

    def move_left_logic(self):
        step_reward = 0 
        new_grid = np.zeros((4, 4), dtype=int)
        moved = False
        
        for r in range(4):
            row = self.grid[r]
            non_zeros = row[row != 0]
            new_row = []
            skip = False
            for i in range(len(non_zeros)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i+1]:
                    doubled = non_zeros[i] * 2
                    new_row.append(doubled)
                    step_reward += doubled
                    skip = True
                else:
                    new_row.append(non_zeros[i])
            
            final_row = np.array(new_row + [0] * (4 - len(new_row)))
            if not np.array_equal(self.grid[r], final_row):
                moved = True
            new_grid[r] = final_row
            
        self.grid = new_grid
        return moved, step_reward

    def step(self, action):
        moved = False
        reward = 0
        
        if action == 0: # Left
            moved, reward = self.move_left_logic()
        elif action == 1: # Up
            self.grid = np.rot90(self.grid, 1)
            moved, reward = self.move_left_logic()
            self.grid = np.rot90(self.grid, -1)
        elif action == 2: # Right
            self.grid = np.rot90(self.grid, 2)
            moved, reward = self.move_left_logic()
            self.grid = np.rot90(self.grid, -2)
        elif action == 3: # Down
            self.grid = np.rot90(self.grid, -1)
            moved, reward = self.move_left_logic()
            self.grid = np.rot90(self.grid, 1)

        if not moved:
            reward = -10
        else:
            self.add_random_tile()
            self.score += reward

        done = self.is_game_over()
        
        return self.grid, reward, done

    def is_game_over(self):
        if len(np.argwhere(self.grid == 0)) > 0: return False
        for r in range(4):
            for c in range(3):
                if self.grid[r,c] == self.grid[r,c+1]: return False
        for r in range(3):
            for c in range(4):
                if self.grid[r,c] == self.grid[r+1,c]: return False
        return True