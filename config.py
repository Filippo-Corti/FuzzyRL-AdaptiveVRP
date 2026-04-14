from typing import Literal

# VRP Instance Generation Config:
NUM_NODES = 10                                                      # Number of customer nodes (excluding depot)     
ENV_DEPOT_MODE: Literal["center", "random"] = "center"              # "center" places depot at (0.5, 0.5), "random" samples depot like other nodes
ENV_NODE_XY_RANGE = (0.0, 1.0)                                      # Range for x and y coordinates of nodes
ENV_DEMAND_RANGE = (1, 3)                                           # Range for customer demand (inclusive)
ENV_CAPACITY_RANGE = (5, 9)                                         # Range for truck capacity (inclusive)

# Agent Configuration
AGENT_MODE: Literal["transformer", "fuzzy"] = "transformer"         # "transformer" or "fuzzy"
CHECKPOINT_TRANSFORMER_PATH = "checkpoints/transformer.pt"          # Path to trained transformer agent checkpoint
CHECKPOINT_FUZZY_PATH = "checkpoints/fuzzy.pkl"                     # Path to trained fuzzy agent checkpoint (pickle file)
SEED = None                                                         # Set to an int for visualizing the same instance

# Fuzzy agent hyperparameters (overridden if loading from checkpoint)
FUZZY_EPSILON = 0.9                                                 # Initial exploration rate for epsilon-greedy action selection
FUZZY_EPSILON_MIN = 0.001                                            # Minimum exploration rate after decay                
FUZZY_EPSILON_DECAY = 0.9995                                         # Multiplicative decay factor for epsilon after each episode
FUZZY_LR = 0.01                                                     # Learning rate for Q-table updates
FUZZY_GAMMA = 0.95                                                  # Discount factor for future rewards in Q-learning updates

# Transformer agent hyperparameters (overridden if loading from checkpoint)
TRANSFORMER_NODE_FEATURES = 4                                       # Number of features for each node (e.g. x, y, demand, visited)
TRANSFORMER_STATE_FEATURES = 3                                      # Number of features for truck state (e.g. x, y, remaining capacity)
TRANSFORMER_D_MODEL = 64                                            # Dimensionality of transformer model embeddings                        
TRANSFORMER_LR = 1e-4                                               # Learning rate for transformer optimizer                       

# Trainer runtime execution settings
TRAINER_BATCH_SIZE = 32                                              # Number of VRP instances to train on in parallel (transformer only)
TRAINER_SAVE_EVERY = 25                                             # Save a checkpoint every N episodes
TRAINER_TORCH_THREADS = 1                                            # Number of CPU threads for PyTorch to use during training   

# Visualization Config:
WINDOW_W = 1920                                                     # Window width in pixels
WINDOW_H = 1080                                                     # Window height in pixels
GRAPH_W = 1360                                                      # Graph width in pixels
HUD_W = WINDOW_W - GRAPH_W                                          # HUD width in pixels
FPS_CAP = 240                                                       # Frame rate limit
FONT_SIZE = 25                                                      # Font size for main text                   
FONT_SIZE_SMALL = int(FONT_SIZE * 0.75)                             # Font size for secondary text
POLL_INTERVAL_S = 2.0                                               # Seconds between automatic simulation resets
DEFAULT_SPEED = 0.01                                                # Default simulation speed        
SPEED_STEP = 0.005                                                  # Amount to increase/decrease speed when adjusting
SPEED_MIN = 0.0001                                                  # Minimum simulation speed 
SPEED_MAX = 1.0                                                     # Maximum simulation speed 
