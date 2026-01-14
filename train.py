import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
import os

# Tvůj nový modul pro Discord
import discord

from engine import GameEngine 
from ai import NeuralNetwork, prepare_grid_for_ai

# --- 1. NASTAVENÍ (HYPERPARAMETERS) ---
BATCH_SIZE = 4096         
LR = 0.0005             
GAMMA = 0.99            
EPSILON_START = 1.0     
EPSILON_END = 0.02      
EPSILON_DECAY = 0.9997  
MEMORY_SIZE = 100000     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

env = GameEngine()
model = NeuralNetwork().to(device) 
optimizer = optim.Adam(model.parameters(), lr=LR) 
criterion = nn.MSELoss() 
memory = deque(maxlen=MEMORY_SIZE) 

epsilon = EPSILON_START
episodes = 0
scores = []

# --- DISCORD SETUP ---
# Sledujeme nejlepší dlaždici této relace, abychom neposílali každou blbost.
# Pokud nechceš notifikace pro malá čísla, dej sem třeba 512.
session_record = 0 

print("Start Training Dojo... (Press Ctrl+C to stop)")

# --- 3. TRÉNINKOVÁ SMYČKA ---
try:
    while True: 
        state = env.reset() 
        state_tensor = prepare_grid_for_ai(state).to(device)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # --- A. VÝBĚR AKCE ---
            if random.random() < epsilon:
                action = random.randint(0, 3) 
            else:
                with torch.no_grad(): 
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            # --- B. PROVEDENÍ TAHU ---
            next_state, reward, done = env.step(action)
            next_state_tensor = prepare_grid_for_ai(next_state).to(device)
            
            # --- C. ULOŽENÍ DO PAMĚTI ---
            memory.append((state_tensor, action, reward, next_state_tensor, done))
            
            state = next_state_tensor
            total_reward += reward
            steps += 1

            # --- D. UČENÍ (REPLAY) ---
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                current_q = model(states).gather(1, actions).squeeze()
                
                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target_q = rewards + (GAMMA * max_next_q * (1 - dones))
                
                loss = criterion(current_q, target_q)
                
                optimizer.zero_grad() 
                loss.backward()       
                optimizer.step()      

        # --- E. KONEC EPIZODY ---
        episodes += 1
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY) 
        scores.append(env.score)

        # --- DISCORD NOTIFIKACE (NOVÉ) ---
        current_max_tile = np.max(env.grid)
        
        # Pokud jsme překonali dosavadní rekord této session, pošli zprávu
        if current_max_tile > session_record:
            session_record = current_max_tile
            print(f"NOVÝ REKORD: {int(session_record)}! Odesílám notifikaci...")
            
            discord_logger.send_new_record_notification(
                tile=session_record,
                episode=episodes,
                epsilon=epsilon,
                score=env.score
            )

        # Výpis každých 100 her
        if episodes % 100 == 0:
            avg_score = sum(scores[-10:]) / 10
            max_tile = np.max(env.grid)
            print(f"Ep: {episodes} | Avg Score: {avg_score:.1f} | Max Tile: {max_tile} | Epsilon: {epsilon:.3f}")
            
        # Uložení modelu každých 500 her
        if episodes % 500 == 0:
            torch.save(model.state_dict(), "model_cnn.pth") # Upravil jsem název na cnn
            print("Model saved!")

except KeyboardInterrupt:
    print("\nTrénink přerušen uživatelem.")
    torch.save(model.state_dict(), "model_cnn_final.pth")
    print("Model uložen.")