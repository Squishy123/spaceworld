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
        if isinstance(self.env.action_space, spaces.Box):
            num_actions = self.env.action_space.shape[0]

        # set agent
        self.agent = agent

        # set config
        self.config = config

        # replay memory
        self.replay_memory = ReplayMemory(self.config['MEMORY_CAPACITY'])

        # model net
        self.model = Transform_Autoencoder(num_actions, frame_stacks=self.config["FRAME_STACK"])
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING_RATE'], weight_decay=self.config['WEIGHT_DECAY'])

        # optimizer for state prediction
        # state_parameters_to_update = list(self.model.encoder.parameters()) + list(self.model.bottleneck.parameters()) + list(self.model.decoder.parameters())
        # self.state_optimizer = torch.optim.Adam(state_parameters_to_update, lr=self.config['LEARNING_RATE'], weight_decay=self.config['WEIGHT_DECAY'])

        # optimizer for reward prediction
        # reward_parameters_to_update = list(self.model.reward_predictor.parameters())
        # self.reward_optimizer = torch.optim.Adam(reward_parameters_to_update, lr=self.config['LEARNING_RATE'], weight_decay=self.config['WEIGHT_DECAY'])

        # sim
        self.screen_stack = []

    # load model weights

    def load(self, path="world_model_weights.pth"):
        checkpoint = torch.load(path, map_location=self.device)

        # reinit
        self.__init__(self.env, self.agent, checkpoint['config'])

        # model net
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.state_optimizer.load_state_dict(checkpoint['state_optimizer_state_dict'])
        # self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])

    # save model weights
    def save(self, path="world_model_weights.pth"):
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def get_screen(self, mod=None):
        # if mod == "start":
        #    return torch.cat((torch.zeros(1, 1, 64, 64, device=self.device), torch.ones(1, 1, 64, 64, device=self.device), torch.zeros(1, 1, 64, 64, device=self.device)), dim=1).to(self.device)

        # if mod == "end":
        #    return torch.cat((torch.ones(1, 1, 64, 64, device=self.device), torch.zeros(1, 1, 64, 64, device=self.device), torch.zeros(1, 1, 64, 64, device=self.device)), dim=1).to(self.device)

        screen = self.env.render(mode='rgb_array')
        screen = screen.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        screen = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Resize(64, interpolation=Image.CUBIC),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(screen).unsqueeze(0).to(self.device)
        return screen

    # optimize model
    def learn(self):
        total_state_loss = np.zeros(self.config["FRAME_DENOISE_SIZE"])
        total_reward_loss = np.zeros(self.config["FRAME_DENOISE_SIZE"])

        if len(self.replay_memory) < self.config['BATCH_SIZE']:
            return total_state_loss, total_reward_loss

        # sample from replaymemory
        batch = self.replay_memory.sample(self.config['BATCH_SIZE'], False)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        total_loss = 0

        for i in range(self.config["FRAME_DENOISE_SIZE"]):
            if i == 0:
                computed_next_state, computed_reward = self.model(state_batch, action_batch)
            else:
                computed_next_state, computed_reward = self.model(computed_next_state[0:self.config['BATCH_SIZE']-i], action_batch[i:self.config['BATCH_SIZE']])
            state_loss = torch.nn.functional.mse_loss(computed_next_state, next_state_batch[i:self.config['BATCH_SIZE']])
            reward_loss = torch.nn.functional.mse_loss(computed_reward.squeeze(1), reward_batch[i:self.config['BATCH_SIZE']])

            total_state_loss[i] = state_loss.item()
            total_reward_loss[i] = reward_loss.item()

            total_loss += state_loss + reward_loss

        total_loss.backward()

        for param in self.model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_state_loss, total_reward_loss

    def reset(self, num_frames_boost=0):
        self.env.reset()
        init_state = self.get_screen(mod="start")
        self.screen_stack = deque([init_state] * self.config['FRAME_STACK'], maxlen=self.config['FRAME_STACK'])

        for _ in range(num_frames_boost):
            init_state, reward, done, _ = self.env.step(0)
            self.screen_stack.append(self.get_screen())
        self.env.close()

        return self.render()

    def render(self):
        return np.clip(self.screen_stack[0].detach().index_select(1, torch.tensor([0, 1, 2], device=self.device)).cpu().squeeze(0).permute(1, 2, 0).numpy(), 0, 1)

    def step(self, action):
        if type(action) is np.ndarray:
            step_action = torch.tensor([], device=self.device)
            for a in action:
                step_action = torch.cat((step_action, torch.empty(1, 1, 10, 10, device=self.device).fill_(a)), dim=1).to(self.device)
        else:
            step_action = torch.empty(1, 1, 10, 10, device=self.device).fill_(action).to(self.device)

        state_batch = torch.cat(tuple(self.screen_stack), dim=1).to(self.device)

        # calculate next state
        computed_next_state, computed_reward = self.model(state_batch, step_action)
        done = False
        if torch.sum(computed_next_state) == torch.tensor(0):
            done = True
        self.screen_stack.extend(torch.split(computed_next_state, self.config["FRAME_STACK"], dim=1))

        return self.render(), computed_reward.item(), done, None

    def no_update_step(self, screen_stack, action):
        if type(action) is np.ndarray:
            step_action = torch.tensor([], device=self.device)
            for a in action:
                step_action = torch.cat((step_action, torch.empty(1, 1, 10, 10, device=self.device).fill_(a)), dim=1).to(self.device)
        else:
            step_action = torch.empty(1, 1, 10, 10, device=self.device).fill_(action).to(self.device)

        state_batch = torch.cat(tuple(screen_stack), dim=1).to(self.device)

        # calculate next state
        computed_next_state, computed_reward = self.model(state_batch, step_action)
        done = False
        if torch.sum(computed_next_state) == torch.tensor(0):
            done = True
        #self.screen_stack.extend(torch.split(computed_next_state, self.config["FRAME_STACK"], dim=1))
        return self.render(), computed_reward.item(), done, None

    # training cycle

    def train(self, callbacks=[], render=False):
        # init callbacks
        for c in callbacks:
            c.init()

        for epoch in range(1, self.config["NUMBER_OF_EPOCHS"] + 1):
            for episode in range(1, self.config["EPISODES_PER_EPOCH"] + 1):
                # reset env
                self.env.reset()
                num_steps = 0
                rewards = 0

                init_state = self.get_screen(mod="start")
                screen_stack = deque([init_state] * self.config['FRAME_STACK'], maxlen=self.config['FRAME_STACK'])
                state = torch.cat(list(screen_stack), dim=1)

                ep_reward = 0
                num_steps = 0
                ep_state_loss = np.zeros(self.config["FRAME_DENOISE_SIZE"])
                ep_reward_loss = np.zeros(self.config["FRAME_DENOISE_SIZE"])

                negative_reward_count = 0

                while True:
                    done = False
                    reward = 0

                    step_reward = 0

                    action = self.agent.act(state, reward, done)
                    if type(action) is np.ndarray:
                        step_action = torch.tensor([], device=self.device)
                        for a in action:
                            step_action = torch.cat((step_action, torch.empty(1, 1, 10, 10, device=self.device).fill_(a)), dim=1)
                    else:
                        step_action = torch.empty(1, 1, 10, 10, device=self.device).fill_(action)
                    # print(step_action.shape)

                    for _ in range(self.config["FRAME_SKIP"]):
                        num_steps += 1
                        next_state, reward, done, _ = self.env.step(action)
                        step_reward += reward
                        if done:
                            break

                    ep_reward += step_reward
                    step_reward = torch.tensor([step_reward], device=self.device).float()

                    # generate next state stack
                    screen_stack.append(self.get_screen())
                    if done:
                        screen_stack.extend(self.config["FRAME_STACK"] * [self.get_screen(mod="end")])

                    next_state = torch.cat(list(screen_stack), dim=1)
                    # print(next_state.shape)
                    # append to replay memory
                    self.replay_memory.append(state, step_action, next_state, step_reward)
                    state = next_state

                    if step_reward < 0 and num_steps > 200:
                        negative_reward_count += 1
                        if negative_reward_count > 30:
                            break
                        else:
                            negative_reward_count = 0

                    state_loss, reward_loss = self.learn()
                    ep_state_loss += state_loss
                    ep_reward_loss += reward_loss

                    if done:
                        break

                # run callbacks
                for c in callbacks:
                    c.callback(self, epoch, episode, ep_reward, (ep_state_loss, ep_reward_loss), num_steps)

        # close callbacks
        for c in callbacks:
            c.close()
