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

rewards = []
num_steps_acc = []
state_losses = []
reward_losses = []


def log(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    print(f"EPOCH: {epoch} - EPISODE: {episode} - REWARD: {ep_reward} - LOSS: {ep_loss} - NUM_STEPS: {num_steps}")


def save(model, epoch, episode, ep_reward, ep_loss, num_steps):
    if episode % 100 == 0:
        print("SAVING MODEL")
        model.save(f"results/world_model_weights_{epoch}_{episode}.pth")
        np.savetxt("results/data.txt", np.array([rewards, num_steps_acc, state_losses, reward_losses]))


fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)
fig3, (ax3) = plt.subplots(1, constrained_layout=True)
fig4, (ax4) = plt.subplots(1, constrained_layout=True)


def print_screen(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    if episode % 10 == 0:
        batch = agent.replay_memory.sample(1)

        state_batch = torch.cat(batch.state).to(agent.device)
        action_batch = torch.cat(batch.action).to(agent.device)
        reward_batch = torch.cat(batch.reward).to(agent.device)
        next_state_batch = torch.cat(batch.next_state).to(agent.device)

        computed_next_state, computed_reward = agent.model(state_batch, action_batch)
        computed_image = computed_next_state[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy()
        state_image = state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy()
        next_state_image = next_state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2])).cpu().squeeze(0).permute(1, 2, 0).numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.clip(state_image, 0, 1))
        ax1.set_title("current-state")
        ax2.imshow(np.clip(computed_image, 0, 1))
        ax2.set_title("predicted-next-state")
        ax3.imshow(np.clip(next_state_image, 0, 1))
        ax3.set_title("next-state")

        # ax1.text(0, 100, str([action_batch[0][0][0][0].item(), action_batch[0][1][0][0].item(), action_batch[0][2][0][0].item()]), fontsize=10)
        ax1.text(0, 100, str(action_batch[0][0][0][0].item()), fontsize=10)
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

    state_loss, reward_loss = ep_loss
    ax2.set_title('State Loss Over Episodes')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.scatter(((epoch-1) * 100) + episode, state_loss, color="red", label="state_loss")
    state_losses.append(state_loss)

    ax3.set_title('Reward Loss Over Episodes')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Loss')
    ax3.scatter(((epoch-1) * 100) + episode, reward_loss, color="red", label="reward_loss")
    reward_losses.append(reward_loss)

    ax4.set_title('Duration Over Episodes')
    ax4.set_ylabel('Duration')
    ax4.set_xlabel('Episodes')
    ax4.scatter(((epoch-1) * 100) + episode, num_steps, color="orange")
    num_steps_acc.append(num_steps)

    fig1.savefig("results/plt1.png")
    fig2.savefig("results/plt2.png")
    fig3.savefig("results/plt3.png")
    fig4.savefig("results/plt4.png")


# print(model.get_screen().shape)
'''
model.load("results_baseline/world_model_weights_5_100.pth")
agent = HumanAgent()

state = model.reset()
reward = 0
done = False

fig5, (ax5) = plt.subplots(1, constrained_layout=True)
ax5.set_title("Loss over Step")
ax5.set_xlabel('Step')
ax5.set_ylabel('Loss')
for i in range(100):
    # print(i)
    action = agent.act(state, reward, done)
    # print(action)
    model.env.step(action)
    env_state = model.env.render()
    for _ in range(1):
        model.step(action)
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
'''
model.train(render=True, callbacks=[log, save, plot, print_screen])  # [log, save, plot])
