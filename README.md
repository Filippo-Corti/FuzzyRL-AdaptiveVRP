# VRP Experimental Framework: Fuzzy RL vs Transformer vs Heuristic

---

This repository studies the **Vehicle Routing Problem** (VRP), 
a classical NP-hard optimization problem concerned with designing 
routes for a fleet of vehicles under operational constraints.

Traditional approaches rely on heuristics and metaheuristics, which are 
intuitive and interpretable but limited by hand-crafted assumptions.

More recent deep learning methods learn routing strategies 
directly from data and achieve strong performance, at the cost of 
reduced interpretability.

This work explores a middle ground between these approaches
by introducing a fuzzy reinforcement learning agent for the VRP. 
A comparison is conducted on a capacitated VRP with clustered
customer distributions and soft time window constraints,
evaluating three policies:
* A **Time-Oriented Nearest Neighbour** (TONN) heuristic;
* A **Transformer-based Policy** trained with *REINFORCE*;
* A **Neuro-Fuzzy Scoring Agent** also trained using *REINFORCE*. 

Results showed that the neuro-fuzzy agent improves upon the heuristic baseline by over 5\%, while retaining a degree of interpretability and remaining competitive with the learned approach.

## Instructions

1) Setup a virtual environment:
  ```bash
  python -m venv .venv
  ```

2) Activate virtual environment:
  ```bash
  source .venv/bin/activate
  ```

3) Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

4) Run any of the entry points to the repository:
  ```bash
  python main.py # Runs visualization
  ```
  ```bash
  python train.py --agent [fuzzy/transformer] # Runs training
  ```
  ```bash
  python compare.py [--tonn] [--fuzzy checkpoints/fuzzy-12000] [--transformer checkpoints/transformer-4000]  # Runs test comparison
  ```

## Repository Structure

The repository is structured as follows:
* [`src/`](src/) contains the implementations for the environment, the agents, the training loops and the visualization interface;
* [`checkpoints/`](checkpoints/) contains pre-trained checkpoints for the REINFORCE-based agents;
* [`scripts/`](scripts/) contains utility scripts for drawing instances, analyzing checkpoints and training curves.