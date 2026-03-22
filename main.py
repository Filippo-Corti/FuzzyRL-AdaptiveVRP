from env import VRPEnvironment
from utils import parse_vrp_instance
from viz import run

graph = parse_vrp_instance(path="assets/datasets/CVRPLIB-Augerat-A/A-n32-k5.vrp")

env = VRPEnvironment(graph=graph)

run(env, None)  # blocks until window is closed
