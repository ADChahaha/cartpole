# DQN算法实现

## DQN优化目标
$$
J(\theta) = \mathbb{E}[[q(s, a) - \hat{q}(s, a,\theta)]^{2}] \\
\arg \min_{\theta}J(\theta)
$$

## 实现方法
梯度下降
$$
\theta_{t+1} = \theta_{t} + \alpha \nabla{J(\theta)} \\
\nabla{J(\theta)} = 2[\hat{q}(s, a, \theta) - q(s, a)]\nabla{\hat{q}(s,a)}
$$
