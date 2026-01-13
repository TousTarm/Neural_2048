import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
import os

from engine import GameEngine 
from ai import NeuralNetwork, prepare_grid_for_ai

# --- 1. NASTAVENÍ (HYPERPARAMETERS) ---
# S těmito čísly si data scientists hrají celé dny.
BATCH_SIZE = 64         # Kolik situací z minulosti si připomene najednou
LR = 0.0005             # Rychlost učení (Learning Rate). Moc velké = AI blázní. Moc malé = AI se učí pomalu.
GAMMA = 0.99            # Jak moc jí záleží na budoucnosti (0.99 = hodně).
EPSILON_START = 1.0     # Na začátku 100% náhoda (průzkum).
EPSILON_END = 0.02      # Na konci 2% náhoda (aby nezakrněla).
EPSILON_DECAY = 0.9997  # Jak rychle klesá náhoda (čím blíže 1, tím pomaleji).
MEMORY_SIZE = 50000     # Kapacita paměti (počet tahů).

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

env = GameEngine()
model = NeuralNetwork().to(device) # Nahrajeme mozek na procesor/grafiku
optimizer = optim.Adam(model.parameters(), lr=LR) # Nástroj, který upravuje váhy v mozku
criterion = nn.MSELoss() # Funkce pro výpočet chyby (Loss)
memory = deque(maxlen=MEMORY_SIZE) # Paměť (Replay Buffer)

epsilon = EPSILON_START
episodes = 0
scores = []

print("Start Training Dojo... (Press Ctrl+C to stop)")

# --- 3. TRÉNINKOVÁ SMYČKA ---
try:
    while True: # Nekonečná smyčka epizod (her)
        state = env.reset() # Reset hry
        # Převedeme grid na tensor pro AI
        state_tensor = prepare_grid_for_ai(state).to(device)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # --- A. VÝBĚR AKCE (Epsilon-Greedy) ---
            # Buď hodíme kostkou (průzkum), nebo použijeme mozek (využití)
            if random.random() < epsilon:
                action = random.randint(0, 3) 
            else:
                with torch.no_grad(): # Tady se neučíme, jen ptáme
                    q_values = model(state_tensor)
                    # argmax vrátí index největšího čísla (nejlepší tah)
                    action = torch.argmax(q_values).item()

            # --- B. PROVEDENÍ TAHU ---
            next_state, reward, done = env.step(action)
            next_state_tensor = prepare_grid_for_ai(next_state).to(device)
            
            # --- C. ULOŽENÍ DO PAMĚTI ---
            # (Co jsem viděl, Co jsem udělal, Co jsem dostal, Co se stalo pak, Konec?)
            memory.append((state_tensor, action, reward, next_state_tensor, done))
            
            state = next_state_tensor
            total_reward += reward
            steps += 1

            # --- D. UČENÍ (REPLAY) ---
            # Začneme se učit, až když máme v paměti aspoň 'BATCH_SIZE' vzpomínek
            if len(memory) > BATCH_SIZE:
                # 1. Náhodný výběr z paměti
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Převod na PyTorch tensory (hromadně)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # 2. Co si mozek myslí TEĎ (Current Q)
                # .gather vybere z výstupu jen tu hodnotu pro akci, kterou jsme reálně udělali
                current_q = model(states).gather(1, actions).squeeze()
                
                # 3. Co by si MĚL myslet (Target Q)
                # Bellmanova rovnice: Odměna + (Gamma * Nejlepší možná budoucnost)
                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target_q = rewards + (GAMMA * max_next_q * (1 - dones))
                
                # 4. Update vah (Backpropagation)
                loss = criterion(current_q, target_q)
                
                optimizer.zero_grad() # Vynulovat staré gradienty
                loss.backward()       # Spočítat nové gradienty (kdo za chybu může)
                optimizer.step()      # Upravit váhy

        # --- E. KONEC EPIZODY ---
        episodes += 1
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY) # Snížíme náhodu
        scores.append(env.score)
        
        # Výpis každých 10 her
        if episodes % 10 == 0:
            avg_score = sum(scores[-10:]) / 10
            max_tile = np.max(env.grid)
            print(f"Ep: {episodes} | Avg Score: {avg_score:.1f} | Max Tile: {max_tile} | Epsilon: {epsilon:.3f}")
            
        # Uložení modelu každých 500 her (pro jistotu)
        if episodes % 500 == 0:
            torch.save(model.state_dict(), "model_2048.pth")
            print("Model saved!")

except KeyboardInterrupt:
    print("\nTrénink přerušen uživatelem.")
    torch.save(model.state_dict(), "model_2048_final.pth")
    print("Model uložen.")