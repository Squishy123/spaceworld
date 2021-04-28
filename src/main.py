from config.baseline import config
import torch

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent

import matplotlib.pyplot as plt
from gym import wrappers
import gym
import numpy as np


env = gym.make('CarRacing-v0')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, config)

rewards = []
num_steps_acc = []
loss = []


def log(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    print(f"EPOCH: {epoch} - EPISODE: {episode} - REWARD: {ep_reward} - LOSS: {ep_loss} - NUM_STEPS: {num_steps}")


def save(model, epoch, episode, ep_reward, ep_loss, num_steps):
    if episode % 100 == 0:
        print("SAVING MODEL")
        model.save(f"results/world_model_weights_{epoch}_{episode}.pth")
        np.savetxt("results/data.txt", np.array([rewards, num_steps_acc, loss]))


fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)
fig3, (ax3) = plt.subplots(1, constrained_layout=True)


def print_screen(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    if episode % 10 == 0:
        batch = agent.replay_memory.sample(1)

        state_batch = torch.cat(batch.state).to(agent.device)
        action_batch = torch.cat(batch.action).to(agent.device)
        reward_batch = torch.cat(batch.reward).to(agent.device)
        next_state_batch = torch.cat(batch.next_state).to(agent.device)

        computed_next_state = agent.model(state_batch, action_batch)
        computed_image = computed_next_state[0].detach().unsqueeze(0).index_select(1, torch.tensor([6, 7, 8])).cpu().squeeze(0).permute(1, 2, 0).numpy()
        state_image = state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy()
        next_state_image = next_state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([6, 7, 8])).cpu().squeeze(0).permute(1, 2, 0).numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.clip(state_image, 0, 1))
        ax1.set_title("current-state")
        ax2.imshow(np.clip(computed_image, 0, 1))
        ax2.set_title("predicted-next-state")
        ax3.imshow(np.clip(next_state_image, 0, 1))
        ax3.set_title("next-state")

        ax1.text(0, 100, str([action_batch[0][0][0][0].item(), action_batch[0][1][0][0].item(), action_batch[0][2][0][0].item()]), fontsize=10)
       # ax1.text(0, 100, str(action_batch[0][0][0][0].item()), fontsize=10)
        ax1.text(0, 130, "Left/Right", fontsize=10)
        ax1.text(40, 130, "Gas", fontsize=10)
        ax1.text(80, 130, "Brake", fontsize=10)

        plt.savefig(f"results/print_screen_{epoch}_{episode}.png")


def plot(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    ax1.set_title('Rewards Over Episodes')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards')
    ax1.scatter(((epoch-1) * 100) + episode, ep_reward, color="blue")
    rewards.append(ep_reward)

    ax2.set_title('Loss Over Episodes')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.scatter(((epoch-1) * 100) + episode, ep_loss, color="red")
    loss.append(ep_loss)

    ax3.set_title('Duration Over Episodes')
    ax3.set_ylabel('Duration')
    ax3.set_xlabel('Episodes')
    ax3.scatter((epoch * 100) + episode, num_steps, color="orange")
    num_steps_acc.append(num_steps)

    fig1.savefig("results/plt1.png")
    fig2.savefig("results/plt2.png")
    fig3.savefig("results/plt3.png")


# print(model.get_screen().shape)
model.train(render=True, callbacks=[log, save, plot, print_screen])  # [log, save, plot])
