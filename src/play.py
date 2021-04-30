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

model.load("results/world_model_weights_7_100.pth")
agent = HumanAgent()
reward = 0
state = model.reset()
done = False

for i in range(100):
    # print(i)
    action = agent.act(state, reward, done)
    # print(action)
    model.env.step(action)
    env_state = model.env.render()
    reward = 0
    for _ in range(1):
        _, r, _ = model.step(action)
    reward+=r
    next_state = model.render()

    # loss = torch.nn.functional.mse_loss(torch.tensor(state), torch.tensor(next_state))
    # print(loss)
    # ax5.scatter(i, loss.item(), color="red")

    plt.imshow((model.render()))
    plt.draw()
    plt.pause(1e-1)

    state = next_state

    # fig5.savefig("results/test_loss.png")

env.close()
plt.close()

