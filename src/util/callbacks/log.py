from .callback import Callback


def log_default(agent, epoch, episode, ep_reward, ep_loss, num_steps):
    print(f"EPOCH: {epoch} - EPISODE: {episode} - REWARD: {ep_reward} - LOSS: {ep_loss} - NUM_STEPS: {num_steps}")


default = Callback(
    lambda self: log_default
)
