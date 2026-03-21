from env import VRPEnvironment
from viz import run

env = VRPEnvironment()

run(env, None)  # blocks until window is closed
