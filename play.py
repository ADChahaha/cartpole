import time
from gym.wrappers import TimeLimit
import torch
import gym
import numpy as np
from torchrl.modules import MLP
import pygame

np.bool8 = np.bool_

def start_game(human_control=True, random_action=False, checkpoint_path='./best.pth', frame_time=0.05):
    env = gym.make("CartPole-v1", render_mode="human")
    env = TimeLimit(env, 200)
    init_env = env.reset()
    state = torch.tensor(init_env[0], dtype=torch.float32).unsqueeze(0)

    class ActionMaker:
        def __init__(self, random_action=True, checkpoint_path='./best.pth'):
            self.random_action = random_action
            if random_action:
                self.action_maker = env.action_space.sample
            else:
                self.action_maker = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
                self.action_maker.load_state_dict(torch.load(checkpoint_path))
                self.action_maker.eval()

        def __call__(self, state):
            if self.random_action:
                return self.action_maker()
            else:
                with torch.no_grad():
                    q_sa = self.action_maker(state)
                    _, action = q_sa.max(dim=1)
                    return action.item()

    action_maker = ActionMaker(random_action, checkpoint_path)

    if human_control:
        print("游戏开始！按 'a' 向左，'d' 向右（无输入则自动保持默认方向）。按 Ctrl+C 可退出。")
        pygame.init()
        pygame.display.init()
        pygame.event.set_allowed([pygame.KEYDOWN])

    step = 0
    last_action = 0  # 默认动作（例如往左）

    while True:
        step += 1
        if human_control:
            action = last_action  # 默认使用上一步动作或默认值
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 0
                    elif event.key == pygame.K_d:
                        action = 1
            last_action = action  # 记录这次动作
        else:
            action = action_maker(state)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            env.close()
            if human_control:
                pygame.quit()
            return step

        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        time.sleep(frame_time)  # 控制游戏节奏
