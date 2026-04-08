"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # TODO: Tune the following hyperparameters
    # Replace the default values with your own choices.
    "lr": 1e-3,            
    "epsilon": 0.03,       
    "target_update": 10,   
    "hidden_dim": 128,     
    
    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}