import numpy as np
from .callback import Callback, Context


def save_init(self):
    self.rewards = []
    self.num_steps_acc = []
    self.state_losses = []
    self.reward_losses = []


def save_close(self):
    self.writer.close()


save_context = Context(save_init, save_close)


def save_cb(self):
    def save_model(model, epoch, episode, ep_reward, ep_loss, num_steps):
        state_loss, reward_loss = ep_loss

        self.rewards.append(ep_reward)
        self.state_losses.append(state_loss)
        self.reward_losses.append(reward_loss)
        self.num_steps_acc.append(num_steps)

        if episode % 10 == 0:
            print("SAVING MODEL")
            model.save(f"results/world_model_weights_{epoch}_{episode}.pth")
            np.save("results/data", np.array([self.rewards, self.num_steps_acc, self.state_losses, self.reward_losses], dtype=object))

    return save_model


save_model = Callback(
    save_cb,
    save_context
)
