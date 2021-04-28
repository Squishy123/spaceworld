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

torch.set_default_tensor_type('torch.cuda.FloatTensor')


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

        # optimizer
        parameters_to_update = []
        for p in self.model.parameters():
            if p.requires_grad == True:
                parameters_to_update.append(p)
        self.optimizer = torch.optim.Adam(parameters_to_update, lr=self.config['LEARNING_RATE'], weight_decay=self.config['WEIGHT_DECAY'])

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
        return T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Resize(64, interpolation=Image.CUBIC),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(screen).unsqueeze(0).to(self.device)

    # optimize model
    def learn(self):
        if len(self.replay_memory) < self.config['BATCH_SIZE']:
            return 0, []

        # sample from replaymemory
        batch = self.replay_memory.sample(self.config['BATCH_SIZE'])

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        computed_next_state = self.model(state_batch, action_batch)
        # print(computed_next_state[0].unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().shape)
        # plt.imshow(computed_next_state[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        # plt.draw()
        # plt.pause(1e-3)

        loss = torch.nn.functional.mse_loss(computed_next_state, next_state_batch)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item(), computed_next_state[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy()

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

                ep_reward = 0
                num_steps = 0
                ep_loss = 0

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
                    step_reward = torch.tensor([reward], device=self.device)

                    # generate next state stack
                    screen_stack.append(self.get_screen())
                    next_state = torch.cat(list(screen_stack), dim=1) if not done else torch.zeros(1, 3*self.config["FRAME_STACK"], 64, 64, device=self.device)
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

                    # learn
                    loss_val, generated = self.learn()
                    # if generated != []:
                    #     plt.imshow(np.clip(generated, 0, 1))
                    #     plt.draw()
                    #     plt.pause(1e-3)
                    ep_loss += loss_val

                    if done:
                        break

                # run callbacks
                for c in callbacks:
                    c(self, epoch, episode, ep_reward, ep_loss, num_steps)
