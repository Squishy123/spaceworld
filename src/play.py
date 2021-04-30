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

import pyglet
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import imageio


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)
'''
model.load("results/world_model_weights_10_100.pth")
reward = 0
state = model.reset()
done = False

win = pyglet.window.Window(width=400, height=400)
keys = pyglet.window.key.KeyStateHandler()
win.push_handlers(keys)
current_frame = pyglet.image.ImageData(400, 400, 'rgb', Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST).tobytes())


def update(dt):
    action = 0
    if keys[pyglet.window.key.A]:
        print("A")
        action = 1
    elif keys[pyglet.window.key.W]:
        print("W")
        action = 2
    elif keys[pyglet.window.key.D]:
        print("D")
        action = 3

    reward = 0
    for _ in range(1):
        _, r, _ = model.step(action)
    reward += r
    model.env.render()
    # print(reward)

    current_frame = pyglet.image.ImageData(400, 400, 'rgb', Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST).tobytes())


@win.event
def on_draw():
    win.clear()
    current_frame.blit(0, 0)


pyglet.clock.schedule_interval(update, 0.1)
pyglet.app.run()
'''
def center_crop(im, new_width, new_height):
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    return im.crop((left, top, right, bottom))


model.load("results/world_model_weights_10_100.pth")
# agent = HumanAgent()
model.env.reset()
state = model.reset()
reward = 0
done = False
env_frame = None
with imageio.get_writer(f"boosted_base.mp4", fps=10) as video:
    for i in range(300):
        base = Image.new('RGB', (1280, 720))
        
        # print(i)
        action = agent.act(state, reward, done)
        # print(action)
        _, _, env_done, _ = model.env.step(action)

        if not env_done:
            env_frame = center_crop(Image.fromarray(model.env.render(mode="rgb_array")), 400, 400)
            
            base.paste(env_frame, (90, 160))
            
            #if i % b == 0:
            #    model.screen_stack.extend(model.config["FRAME_STACK"] * [model.get_screen()])

        #env_state = model.env.render()
        reward = 0
        for _ in range(1):
            _, reward, _ = model.step(action)
        next_state = model.render()
        # print(model.get_screen(mod="end").squeeze(0).squeeze(0).cpu().shape)
        #video.append_data(model.get_screen(mod="end").squeeze(0).cpu().permute(1, 2, 0).numpy())
        model_frame = Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST)
        base.paste(model_frame, (790, 160))
        video.append_data(np.array(base))
        state = next_state
        video.append_data(np.array(base))

        # draw final frames
        if env_done:
            for _ in range(100):
                _, reward, _ = model.step(action)
                
                base = Image.new('RGB', (1280, 720))
                model_frame = Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST)
                base.paste(model_frame, (790, 160))
                base.paste(env_frame, (90, 160))
                video.append_data(np.array(base))
            break

