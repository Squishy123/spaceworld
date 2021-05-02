import numpy as np

rewards = []
num_steps_acc = []
state_losses = []
reward_losses = []


def save_model(model, epoch, episode, ep_reward, ep_loss, num_steps):
    state_loss, reward_loss = ep_loss

    rewards.append(ep_reward)
    state_losses.append(state_loss)
    reward_losses.append(reward_loss)
    num_steps_acc.append(num_steps)

    if episode % 100 == 0:
        print("SAVING MODEL")
        model.save(f"results/world_model_weights_{epoch}_{episode}.pth")
        np.savetxt("results/data.txt", np.array([rewards, num_steps_acc, state_losses, reward_losses]))
