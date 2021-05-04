from config.baseline import config as base_config
from config.single_frame import config as sf_config

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.greedy_agent import GreedyAgent

from util.callbacks import log, plot, save, tensorboard
from util.generate_video import generate_world_video


from gym import wrappers
import gym


env = gym.make('LunarLander-v2')

model = World_Model(env, RandomAgent(env.action_space), base_config)

model.load("results_base/world_model_weights_10_100.pth")
model.config = base_config

agent = GreedyAgent(env.action_space, model, base_config)

generate_world_video(model, agent, "test_greedy")
