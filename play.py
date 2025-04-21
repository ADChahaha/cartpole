from torchrl.modules import MLP
import torch
import gym
import numpy as np

np.bool8 = np.bool_

env = gym.make("CartPole-v1", render_mode="human")

init_env = env.reset()
state = torch.tensor(init_env[0])
state.unsqueeze_(dim=0)

model = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
model.load_state_dict(torch.load(f="weight99.pth"))
model.eval()


for step in range(0, 10000):
    q_sa = model(state)
    _, action = q_sa.max(dim=1)
    action = action.item()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        if truncated:
            print("success!")
        else:
            print("failed")
            print(f"step: {step}")
        break
    state = torch.tensor(obs)
    state.unsqueeze_(dim=0)

env.close()
