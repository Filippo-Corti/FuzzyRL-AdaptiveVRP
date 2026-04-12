from typing import Literal

# VRP Instance Generation Config:
NUM_NODES = 12                                                      # Number of customer nodes (excluding depot)     
ENV_DEPOT_MODE: Literal["center", "random"] = "random"              # "center" places depot at (0.5, 0.5), "random" samples depot like other nodes
ENV_NODE_XY_RANGE = (0.0, 1.0)                                      # Range for x and y coordinates of nodes
ENV_DEMAND_RANGE = (1, 1)                                           # Range for customer demand (inclusive)
ENV_CAPACITY_RANGE = (3, 7)                                         # Range for truck capacity (inclusive)

# Agent Configuration
AGENT_MODE = "transformer"                                          # "transformer" or "fuzzy"
CHECKPOINT_TRANSFORMER_PATH = "checkpoints/transformer.pt"          # Path to trained transformer agent checkpoint
CHECKPOINT_FUZZY_PATH = "checkpoints/fuzzy.pkl"                     # Path to trained fuzzy agent checkpoint (pickle file)
SEED = None                                                         # Set to an int for visualizing the same instance

# Trainer runtime execution settings
TRAINER_BATCH_SIZE = 128                                             # Number of VRP instances to train on in parallel
TRAINER_SAVE_EVERY = 100                                             # Save a checkpoint every N episodes
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
SPEED_STEP = 0.01                                                   # Amount to increase/decrease speed when adjusting
SPEED_MIN = 0.0001                                                # Minimum simulation speed 
SPEED_MAX = 1.0                                                     # Maximum simulation speed 
