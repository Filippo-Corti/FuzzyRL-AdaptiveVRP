from typing import Literal

# VRP Instance Generation Config:
NUM_NODES = 30                                                      # Number of customer nodes (excluding depot)
TESTSET_BATCH_SIZE = 200                                            # Number of instances to generate for the custom test set
ENV_DEPOT_MODE: Literal["center", "random"] = "center"              # "center" places depot at (0.5, 0.5), "random" samples depot like other nodes
ENV_NODE_XY_RANGE = (0.0, 1.0)                                      # Range for x and y coordinates of nodes
ENV_WEIGHT_RANGE = (1.0, 5.0)                                       # Range for customer weights
ENV_W_FIXED = 15.0                                                  # Fixed truck capacity W
ENV_INITIAL_VISIBLE_RATIO = 1.0                                     # Fraction of customers visible at timestep 0
ENV_WINDOW_LENGTH_RANGE = (30, 50)                                   # Inclusive time-window length range
ENV_CLUSTER_COUNT_RANGE = (3, 5)                                    # Number of clusters sampled per instance
ENV_OUTLIER_COUNT_RANGE = (2, 7)                                   # Number of outlier customers per instance
ENV_CLUSTER_STD_RANGE = (0.05, 0.14)                                # Cluster std-dev range in normalized coordinates
CUSTOM_TESTSET_PATH = "datasets/custom/vrp_testset_200_n50.pt"      # Output path for generated custom test set
CUSTOM_TONN_RESULTS_PATH = "datasets/custom/tonn_distance_results.pt" # Output path for TONN distance-only vs urgency-only results

# NUM_NODES = 20                                                      # Number of customer nodes (excluding depot)
# TESTSET_BATCH_SIZE = 200                                            # Number of instances to generate for the custom test set
# ENV_DEPOT_MODE: Literal["center", "random"] = "center"              # "center" places depot at (0.5, 0.5), "random" samples depot like other nodes
# ENV_NODE_XY_RANGE = (0.0, 1.0)                                      # Range for x and y coordinates of nodes
# ENV_WEIGHT_RANGE = (1.0, 5.0)                                       # Range for customer weights
# ENV_W_FIXED = 7.0                                                  # Fixed truck capacity W
# ENV_INITIAL_VISIBLE_RATIO = 1.0                                     # Fraction of customers visible at timestep 0
# ENV_WINDOW_LENGTH_RANGE = (5, 7)                                   # Inclusive time-window length range
# ENV_CLUSTER_COUNT_RANGE = (2, 3)                                    # Number of clusters sampled per instance
# ENV_OUTLIER_COUNT_RANGE = (1, 3)                                   # Number of outlier customers per instance
# ENV_CLUSTER_STD_RANGE = (0.05, 0.14)                                # Cluster std-dev range in normalized coordinates
# CUSTOM_TESTSET_PATH = "datasets/custom/vrp_testset_200_n10.pt"      # Output path for generated custom test set
# CUSTOM_TONN_RESULTS_PATH = "datasets/custom/tonn_distance_results.pt" # Output path for TONN distance-only vs urgency-only results


# Agent Configuration
AGENT_MODE: Literal["transformer", "fuzzy"] = "transformer"         # "transformer" or "fuzzy"
CHECKPOINT_TRANSFORMER_PATH = "checkpoints/transformer.pt"          # Path to trained transformer agent checkpoint
CHECKPOINT_FUZZY_PATH = "checkpoints/fuzzy-3550.pkl"                     # Path to trained fuzzy agent checkpoint (pickle file)
SEED = None                                                         # Set to an int for visualizing the same instance

# Fuzzy agent hyperparameters (overridden if loading from checkpoint)
FUZZY_BATCH_SIZE = 512                                              # Number of VRP instances to train on in parallel (fuzzy agent only)
FUZZY_EPSILON = 0.9                                                 # Initial exploration rate for epsilon-greedy action selection
FUZZY_EPSILON_MIN = 0.001                                            # Minimum exploration rate after decay                
FUZZY_EPSILON_DECAY = 0.9995                                         # Multiplicative decay factor for epsilon after each episode
FUZZY_LR = 1e-3                                                     # Learning rate for Q-table updates
FUZZY_GAMMA = 0.95                                                  # Discount factor for future rewards in Q-learning updates

# Transformer agent hyperparameters (overridden if loading from checkpoint)
TRANSFORMER_NODE_FEATURES = 6                                       # Number of node features from env: [x, y, demand_norm, urgency, visited, is_depot]
TRANSFORMER_STATE_FEATURES = 4                                      # Number of truck-state features from env: [x, y, remaining_cap_norm, at_depot]
TRANSFORMER_D_MODEL = 128                                           # Dimensionality of transformer model embeddings                        
TRANSFORMER_LR = 1e-4                                               # Learning rate for transformer optimizer                       

# Trainer runtime execution settings
TRAINER_BATCH_SIZE = 256                                              # Number of VRP instances to train on in parallel (transformer only)
TRAINER_EPISODES = 5000                                               # Number of transformer training episodes
TRAINER_NUM_NODES = NUM_NODES                                        # Number of nodes used during training instance generation
TRAINER_LATENESS_ALPHA = 0.2                                         # Alpha in combined cost C = distance + alpha * lateness
TRAINER_GRAD_CLIP_NORM = 1.0                                         # Gradient clipping norm for REINFORCE updates
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
