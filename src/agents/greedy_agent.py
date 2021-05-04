from gym import spaces
import numpy as np

# RL agent that uses its imagination (dreamed predicted states) to select the best action in a given time step. The predicted states are the results of a pretrained encoder/decoder that is able to approximate the expected pixel representation of a future state given an input pixel space and an action.


class GreedyAgent(object):

    # each Agent has an imagination and a given action space
    def __init__(self, action_space, imagination_model, config):
        self.action_space = []
        if isinstance(action_space, spaces.Discrete):
            self.action_space = [i for i in range(action_space.n)]

        # if isinstance(self.env.action_space, spaces.Box):
        #    num_actions = self.env.action_space.shape[0]

        self.imagination = imagination_model
        self.config = config

    # in order for the agent to intelligently make a decision about what to do right now, he uses his imagination to predict what will happen for each available action to him. "If I go left what will happen?" etc. Following a greedy mindset he takes the best action right now.
    def act(self, observation, reward, done):
        imagined_rewards = {}
        for action in self.action_space:
            # query the imagination
            imagined_image, imagined_reward, done, _ = self.imagination.no_update_step(observation, action)
            imagined_rewards[action] = imagined_reward
        print(imagined_rewards)
        # get the action that had the highest reward as an imagined outcome
        return min(imagined_rewards, key=imagined_rewards.get)

    # to be even smarter this agent can explore imagination trees, where each node is the result of a step through his imagination. The edges of this tree can be considered the various actions available to the agent. Initial trial of tree depth 2
    def smarter_act(self, observation, reward, done):

        # save imagination model starting state
        imagination_root = self.imagination.model.screen_stack
        imagined_rewards = []
        for action in self.action_space:
            total_reward = 0
            for second_action in self.action_space:
                # query the imagination
                imagined_image, imagined_reward = self.imagination.step(action)
                total_reward += imagined_reward
                second_imagined_image, second_imagined_reward = self.imagination.step(second_action)
                total_reward += second_imagined_reward
                # reset imagination to starting state
                self.imagination.model.screen_stack = imagination_root

            imagined_rewards[total_reward] = action

        return imagined_rewards[max(list(imagined_rewards.keys()))]

    def get_screen(self):
        return self.imagination.render()
