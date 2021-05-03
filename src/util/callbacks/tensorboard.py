from torch.utils.tensorboard import SummaryWriter
from .callback import Callback, Context
import matplotlib.pyplot as plt
import numpy as np
import torch
import io


def board_init(self):
    self.writer = SummaryWriter(log_dir="results/runs")


def board_close(self):
    self.writer.close()


board_context = Context(board_init, board_close)


def loss_cb(self):
    def write_loss(agent, epoch, episode, ep_reward, ep_loss, num_steps):
        state_loss, reward_loss = ep_loss

        self.writer.add_scalars("Loss/State_Predictor", {f'{v} Concurrent Frames': k for v, k in enumerate(state_loss)}, ((epoch-1) * 100) + episode)
        self.writer.add_scalars("Loss/Reward_Predictor", {f'{v} Concurrent Frames': k for v, k in enumerate(reward_loss)}, ((epoch-1) * 100) + episode)

    return write_loss


plot_loss = Callback(
    loss_cb,
    board_context
)


def pred_cb(self):
    def write_state(agent, epoch, episode, ep_reward, ep_loss, num_steps):
        if episode % 10 != 0:
            return

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
        '''
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=200)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        '''
        fig.canvas.draw()
        img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.writer.add_image("State/Reward Prediction", img_arr, ((epoch-1) * 100) + episode, dataformats="HWC")

    return write_state


plot_prediction = Callback(
    pred_cb,
    board_context
)
