import torch
from torchrl.modules import MLP
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm
import numpy as np
import torch.utils.tensorboard as tensorboard
# 导入meanmetrics
from torchmetrics import MeanMetric

gamma = 0.99
epochs = 600
# epochs = 1240
batch_size = 64
batch_length = 10
C = 40
step = 0


logger = tensorboard.writer.SummaryWriter()

# buffer = ReplayBuffer(batch_size=batch_size, storage=LazyTensorStorage(max_size=50000, device="cuda"))

# for data in data_collector:
#     buffer.extend(data)

# 模型
model = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
model.load_state_dict(torch.load('weight99.pth'))
target_model = MLP(in_features=4, out_features=2, depth=3, num_cells=64)
target_model.requires_grad_(False)
target_model.load_state_dict(model.state_dict())
model.to(device="cuda")
target_model.to(device="cuda")
# 损失函数
criterion = torch.nn.MSELoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
schduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50, T_mult=2, eta_min=1e-9)

#
env = GymEnv("CartPole-v1", device="cuda")

def policy(tensordict : TensorDict):
    global step
    step += 1
    state = tensordict['observation']
    state = state.unsqueeze(dim=0)
    epsilon = 0.1 + (0.5 - 0.1) * (0.99 ** step)
    with torch.no_grad():
        q_value = target_model(state) # (batch, 2)
        _, action = q_value.max(dim=1)
        action = action if np.random.rand() > epsilon else 1 - action
        action = torch.nn.functional.one_hot(action.squeeze(dim=0), num_classes=2)
    return TensorDict({
        "action": action
        # "action": env.action_spec.sample()
    })

data_collector = SyncDataCollector(env, policy, frames_per_batch=batch_size, total_frames=-1)

# 训练
for i in range(epochs):
    print(f"epoch: {i + 1}")
    progress_bar = tqdm(data_collector,desc="batch")
    avg_loss = MeanMetric().to('cuda')
    avg_diff = MeanMetric().to('cuda')
    avg_q_value = MeanMetric().to('cuda')
    avg_target_q_value = MeanMetric().to('cuda')

    for j, data in enumerate(progress_bar):
        if j >= batch_length:
            break 
        observation = data['observation'] 
        action = data['action']
        # print(action)
        reward = data['next']['reward'] / 10.0
        next_observation = data['next']['observation']
        done = data['next']['done']
        #前向传播
        model.train()
        q_value = model(observation)
        # next_q_value = model(next_observation)
        next_q_value = target_model(next_observation)
        #计算loss
        q_sa = (q_value * action).sum(dim=1).unsqueeze(1)
        max_next_q_value, _ = next_q_value.max(dim=1)
        max_next_q_value.unsqueeze_(dim=1)
        max_current_q_value = reward + gamma * max_next_q_value * (1 - done.float())
        # print(f"max: {max_current_q_value}")
        # print(f"q_sa: {q_sa}")
        loss = criterion(max_current_q_value, q_sa)
        avg_loss.update(loss.item())
        avg_q_value.update(q_sa[0])
        avg_target_q_value.update(max_current_q_value[0])
        avg_diff.update(abs(q_sa[0] - max_current_q_value[0]))
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % C == 0:
        target_model.load_state_dict(model.state_dict())
        torch.save(model.state_dict(), f="weight99.pth")
    logger.add_scalar("loss", avg_loss.compute(), i)
    logger.add_scalar("lr", optimizer.param_groups[0]['lr'], i)
    logger.add_scalar("q_value", avg_q_value.compute(), i)
    logger.add_scalar("target_q_sa", avg_target_q_value.compute(), i)
    logger.add_scalar("diff", avg_diff.compute(), i)
    schduler.step(i)



 
