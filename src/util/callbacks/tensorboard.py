from torch.utils.tensorboard import SummaryWriter
from .callback import Callback


def loss_init(self):
    self.writer = SummaryWriter()


def loss_cb(self):
    def write_loss(agent, epoch, episode, ep_reward, ep_loss, num_steps):
        state_loss, reward_loss = ep_loss
        self.writer.add_scalar("Loss/State_Predictor", state_loss, ((epoch-1) * 100) + episode)
        self.writer.add_scalar("Loss/Reward_Predictor", reward_loss, ((epoch-1) * 100) + episode)

    return write_loss


def loss_close(self):
    self.writer.close()


plot_loss = Callback(
    loss_cb,
    loss_init,
    loss_close
)
