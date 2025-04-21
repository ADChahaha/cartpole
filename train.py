import torch
from torchrl.modules import MLP
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

gamma = 0.99
epochs = 150
batch_size = 128
batch_length = 50
C = 1000
cnt = 0



# buffer = ReplayBuffer(batch_size=batch_size, storage=LazyTensorStorage(max_size=50000, device="cuda"))

# for data in data_collector:
#     buffer.extend(data)

# 模型
model = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
target_model = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
target_model.requires_grad_(False)
target_model.load_state_dict(model.state_dict())
model.to(device="cuda")
target_model.to(device="cuda")
# 损失函数
criterion = torch.nn.MSELoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
schduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50, T_mult=2, eta_min=1e-9)
#
env = GymEnv("CartPole-v1", device="cuda")

def policy(tensordict : TensorDict):
    state = tensordict['observation']
    return TensorDict({
        "action": env.action_spec.sample()
    })


data_collector = SyncDataCollector(env, policy, frames_per_batch=batch_size, total_frames=-1)
# 训练
for i in range(epochs):
    print(f"epoch: {i + 1}")
    progress_bar = tqdm(data_collector,desc="batch")
    for i, data in enumerate(progress_bar):
        if i >= batch_length:
            break 
        observation = data['observation'] 
        action = data['action']
        # print(action)
        reward = data['next']['reward']
        next_observation = data['next']['observation']
        done = data['next']['done']
        #前向传播
        model.train()
        q_value = model(observation)
        next_q_value = target_model(next_observation)
        #计算loss
        q_sa = (q_value * action).sum(dim=1).unsqueeze(1)
        max_next_q_value, _ = next_q_value.max(dim=1)
        max_next_q_value.unsqueeze_(dim=1)
        max_current_q_value = reward + gamma * max_next_q_value * (1 - done.float())
        # print(f"max: {max_current_q_value}")
        # print(f"q_sa: {q_sa}")
        loss = criterion(max_current_q_value, q_sa)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        if cnt > C:
            target_model.load_state_dict(model.state_dict())
            print(loss.item())
            cnt = 0
    schduler.step(i)

torch.save(model.state_dict(), f="weight99.pth")


 
