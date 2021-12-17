from config.baseline import config as base_config
from config.single_frame import config as sf_config
import torch

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent

import matplotlib.pyplot as plt
from gym import wrappers
import gym
import numpy as np


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)

model.load("results_base/world_model_weights_10_100.pth")
agent = HumanAgent()
state = model.reset(40)
reward = 0
done = False

model_frames = []

while True:
    action = agent.act(state, reward, done)
    if action == None:
        break

    state, reward, done, _ = model.step(action)

    model_frames.append(model.render())
    plt.imshow(model.render())
    plt.draw()
    plt.pause(1e-5)

    if done:
        break

fig, (ax) = plt.subplots(1, 5)
ax[1].set_title(f"Simulation Test")
for i in range(5):
    # ax[0][i].imshow(env_frames[i*(len(env_frames)//5)])
    ax[i].imshow(model_frames[i*(len(model_frames)//5)])

fig.savefig(f"sim_test_out.png")
