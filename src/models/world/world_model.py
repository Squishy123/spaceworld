from .transform_autoencoder import Transform_Autoencoder
from util.replay_memory import ReplayMemory

import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gym import spaces
from collections import deque

'''
World Model Class
'''


class World_Model():
    def __init__(self, env, agent, config):
        # set cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set env
        self.env = env
        self.env.reset()

        # set number of transformative actions
        # TODO: Add support for non-discrete spaces aka. box...
        num_actions = 0
        if isinstance(self.env.action_space, spaces.Discrete):
            num_actions = 1

        # set agent
        self.agent = agent

        # set config
        self.config = config

        # replay memory
        self.replay_memory = ReplayMemory(self.config['MEMORY_CAPACITY'])

        # model net
        self.model = Transform_Autoencoder(num_actions)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING_RATE'], weight_decay=self.config['WEIGHT_DECAY'])

    # load model weights
    def load(self, path="world_model_weights.pth"):
        checkpoint = torch.load(path)
        # reinit
        self.__init__(self.env, checkpoint['config'])

        # model net
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # save model weights
    def save(self, path="world_model_weights.pth"):
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        screen = screen.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return T.Compose([T.ToPILImage(),
                          T.Resize(40, interpolation=Image.CUBIC),
                          T.ToTensor()])(screen).unsqueeze(0).to(self.device)

    # optimize model
    def learn(self):
        return
        # training cycle

    def train(self, callbacks=[], render=False):
        for epoch in range(1, self.config["NUMBER_OF_EPOCHS"] + 1):
            for episode in range(1, self.config["EPISODES_PER_EPOCH"] + 1):
                # reset env
                self.env.reset()
                num_steps = 0
                rewards = 0

                init_state = self.get_screen()
                screen_stack = deque([init_state] * self.config['FRAME_STACK'], maxlen=self.config['FRAME_STACK'])
                state = torch.cat(list(screen_stack), dim=1)

                negative_reward_count = 0

                while True:
                    done = False
                    reward = 0

                    # if render == True:
                    #    plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                    #   plt.draw()
                    #  plt.pause(1e-3)

                    step_reward = 0

                    action = self.agent.act(state, reward, done)

                    for _ in range(self.config["FRAME_SKIP"]):
                        num_steps += 1
                        next_state, reward, done, _ = self.env.step(action)
                        step_reward += reward
                        if done:
                            break

                    step_reward = torch.tensor([reward], device=self.device)

                    # generate next state stack
                    screen_stack.append(self.get_screen())
                    next_state = torch.cat(list(screen_stack), dim=1) if not done else None

                    # append to replay memory
                    self.replay_memory.append(state, action, next_state, step_reward)
                    state = next_state

                    if step_reward < 0 and num_steps > 200:
                        negative_reward_count += 1
                        if negative_reward_count > 30:
                            break
                        else:
                            negative_reward_count = 0

                    if done:
                        break
