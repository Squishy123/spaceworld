from config.baseline import config

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent

import matplotlib.pyplot as plt
from gym import wrappers
import gym


env = gym.make('LunarLander-v2')
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
        np.savetxt("data.txt", np.array([rewards, num_steps_acc, loss]))


'''
fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)
fig3, (ax3) = plt.subplots(1, constrained_layout=True)
'''


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
model.train(render=True, callbacks=[log])  # [log, save, plot])
