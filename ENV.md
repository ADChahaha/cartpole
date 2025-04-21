# Env介绍

## TensorDict
TensorDict类是一个增强版的python dict

**`import:`**
```python
from tensordict import TensorDict
```

**`parameters:`**
- `fields` 字典，包含需要使用的obs，action，reward，done， terminated， truncated等，可直接用`[]`得到内部元素 
- `batch_size` 所有字典元素的batch_size
- `device` 所有字典元素的device
- `is_shared`

Example:

```python
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
```



## 环境 Env

**`env.reset()`**
- `input` tensordict类型，可指定reset，可为空
- `output` tensordict类型

**`env.step()`**
- `input` tensordict类型， 必须有action键
- `output` tensordict类型 

Example:

```python
from torchrl.envs import GymEnv
env = GymEnv("CartPole-v1", device='cuda')
reseted_env = env.reset()
print(reseted_env)
>>>
TensorDict(
    fields={
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=cuda:0,
    is_shared=False)
```

```python
action = env.action_spec.rand()
print(env.step(action))
>>> 
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cuda:0,
            is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
```

**`env.rollout()`**

生成一个trajectory
- input 
    - max_step
    - policy 一个接收tensordict，返回tensordict的函数

- output 
    - trajectory 一个tensordict对象

## TransformedEnv
`import`
```python
from torchrl.envs.transforms import TransformedEnv
```

`example`
```python
from torch.envs import GymEnv
from torchrl.envs.transforms import TransformedEnv, RewardScaling
env = GymEnv("CartPole-v1")
transformed_env = TransformedEnv(env, RewardScaling(in_keys=["reward"], scaling=0.1))
```


## Experience Replay

### LazyTensorStorage
`import`
```python
from torch.data import LazyTensorStorage
```
### ReplayBuffer

存储replay的buffer

`import`
```python
from torchrl.data import ReplayBuffer
```
`添加元素`
```python
buffer.add() # 单个元素
buffer.extend() # 多个元素
```

`example`
```python
size = 1000
buffer = ReplayBuffer(storage=LazyTensorStorage(size))
data = env.rollout(max_step) # 假设env已经创建
buffer.extend(data) # 将data展开到buffer内
sample = buffer.sample(sample_size)
print(sample)
>>>
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([5, 2]), device=cpu, dtype=torch.int64, is_shared=False),
        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False),
        observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([5]),
    device=cpu,
    is_shared=False)
```

## 可视化
```python
from torchrl.recode import VideoRecoder
from torchrl.recode.loggers.csv import CSVLogger

env = GymEnv("CartPole-v1", from_pixels=True) # from_pixels要是True，不然不会返回pixels字典
logger = CSVLogger("path", video_format="mp4")
env = TransformedEnv(env, VideoRecoder(logger, tag="video_name"))
env.rollout(max_step)

env.transform.dump() # 这一行是保存视频的代码 
```





