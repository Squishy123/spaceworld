from config.baseline import config as base_config
from config.single_frame import config as sf_config

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent

from util.callbacks import log, plot, save, tensorboard


from gym import wrappers
import gym


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)

model.load("results_base/world_model_weights_10_100.pth")
model.config = base_config

model.train(render=True, callbacks=[log.default, save.save_model, plot.plot_general, plot.plot_state, tensorboard.plot_loss,
            tensorboard.plot_prediction])  # [log.default, save.save_model, plot.plot_general, plot.display_state])
