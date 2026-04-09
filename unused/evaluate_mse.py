import numpy as np
from solver import solve

def evaluate_mse(true_alpha, agent_alpha, times):
    print(f"Comparing MSE over grid between True Alpha ({true_alpha}) and Agent Alpha ({agent_alpha})\n")
    
    for t in times:
        # Get true target profile at time t
        x, T_true = solve(alpha=true_alpha, t_final=t)
        
        # Get agent's predicted profile at time t
        _, T_agent = solve(alpha=agent_alpha, t_final=t)
        
        # Compute MSE
        mse = np.mean((T_agent - T_true)**2)
        print(f"Time T={t}s : MSE = {mse:.6e}")
