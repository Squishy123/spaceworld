import matplotlib.pyplot as plt
import numpy as np
import torch

from .callback import Callback

fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)
fig3, (ax3) = plt.subplots(1, constrained_layout=True)
fig4, (ax4) = plt.subplots(1, constrained_layout=True)


def plot_general(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    ax1.set_title('Rewards Over Episodes')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards')
    ax1.scatter(((epoch-1) * 100) + episode, ep_reward, color="blue")

    state_loss, reward_loss = ep_loss
    ax2.set_title('State Loss Over Episodes')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.scatter(((epoch-1) * 100) + episode, state_loss, color="red", label="state_loss")

    ax3.set_title('Reward Loss Over Episodes')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Loss')
    ax3.scatter(((epoch-1) * 100) + episode, reward_loss, color="red", label="reward_loss")

    ax4.set_title('Duration Over Episodes')
    ax4.set_ylabel('Duration')
    ax4.set_xlabel('Episodes')
    ax4.scatter(((epoch-1) * 100) + episode, num_steps, color="orange")

    fig1.savefig("results/plt1.png")
    fig2.savefig("results/plt2.png")
    fig3.savefig("results/plt3.png")
    fig4.savefig("results/plt4.png")


plot_general = Callback(lambda self: plot_general)


def display_state(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    if episode % 10 == 0:
        batch = agent.replay_memory.sample(1)

        state_batch = torch.cat(batch.state).to(agent.device)
        action_batch = torch.cat(batch.action).to(agent.device)
        reward_batch = torch.cat(batch.reward).to(agent.device)
        next_state_batch = torch.cat(batch.next_state).to(agent.device)

        computed_next_state, computed_reward = agent.model(state_batch, action_batch)
        computed_image = computed_next_state[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2], device=agent.device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        state_image = state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2], device=agent.device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        next_state_image = next_state_batch[0].detach().unsqueeze(0).index_select(1, torch.tensor([0, 1, 2], device=agent.device)).squeeze(0).permute(1, 2, 0).cpu().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.clip(state_image, 0, 1))
        ax1.set_title("current-state")
        ax2.imshow(np.clip(computed_image, 0, 1))
        ax2.set_title("predicted-next-state")
        ax3.imshow(np.clip(next_state_image, 0, 1))
        ax3.set_title("next-state")

        # ax1.text(0, 100, str([action_batch[0][0][0][0].item(), action_batch[0][1][0][0].item(), action_batch[0][2][0][0].item()]), fontsize=10)
        ax1.text(0, 100, "Action Taken - ", fontsize=10)
        action = "Do Nothing"
        if action_batch[0][0][0][0].item() == 1:
            action = "Left"
        elif action_batch[0][0][0][0].item() == 2:
            action = "Up"
        elif action_batch[0][0][0][0].item() == 3:
            action = "Right"
        ax1.text(0, 110, action, fontsize=10)

        ax1.text(80, 100, "Computed Reward", fontsize=10)
        ax1.text(80, 110, "{:.2f}".format(computed_reward.item()), fontsize=10)

        ax1.text(160, 100, "Actual Reward", fontsize=10)
        ax1.text(160, 110, "{:.2f}".format(reward_batch[0].item()), fontsize=10)

        plt.savefig(f"results/print_screen_{epoch}_{episode}.png")


plot_state = Callback(lambda self: display_state)
