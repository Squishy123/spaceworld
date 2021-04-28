from config.baseline import config

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent

import gym


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, config)

model.train(render=True)
