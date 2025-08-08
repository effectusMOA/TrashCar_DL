import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

## 1. 환경: WasteCollectionEnv
class WasteCollectionEnv:
    """
    쓰레기 수거 시뮬레이션 환경.
    목표: 모든 쓰레기통의 채움 수준을 특정 임계값(30%) 이하로 유지.
    """
    def __init__(self, num_bins=6):
        self.num_bins = num_bins
        # 쓰레기통 좌표
        self.bin_coordinates = np.array([
            [0.5, 0.5], [1.5, 1.0], [4.5, 0.5],
            [2.5, 3.0], [1.0, 4.0], [4.0, 4.5]
        ])
        # 인구 밀도: 높을수록 쓰레기 증가 속도 빠름
        self.population_density = np.array([0.1, 0.3, 0.1, 0.5, 0.2, 0.4])

        # 보상 가중치 (Optuna 튜닝)
        self.reward_weight = 0.1313267003485515
        self.overflow_penalty_weight = -10.42169302529806

        self.action_space_size = num_bins
        self.state_space_size = 1 + num_bins
        self.start_bin_index = 0
        self.max_steps_per_episode = num_bins * 3 # 에피소드 당 최대 스텝

        # 성공 조건
        self.success_threshold = 30.0 # 모든 쓰레기통이 이 값 이하가 되면 성공
        self.success_bonus = 50.0     # 성공 보너스

    def reset(self):
        # 환경 초기화
        self.bin_fill_levels = np.random.uniform(0, 70, size=self.num_bins)
        self.bin_fill_levels[self.start_bin_index] = 0 # 시작점은 비움
        self.current_bin_index = self.start_bin_index
        self.steps_taken = 0
        state = np.concatenate(([self.current_bin_index], self.bin_fill_levels / 100.0))
        return state

    def step(self, action):
        # 액션 수행 및 상태 업데이트
        if not (0 <= action < self.num_bins):
            raise ValueError(f"Invalid action: {action}")

        target_bin_index = action
        distance = np.linalg.norm(self.bin_coordinates[self.current_bin_index] - self.bin_coordinates[target_bin_index])
        time_passed = distance
        collected_trash = self.bin_fill_levels[target_bin_index]

        # 보상 계산
        reward = (collected_trash * self.reward_weight) - distance

        # 패널티
        if collected_trash < 10: # 적은 양의 쓰레기 수거 패널티
            reward -= 5
        if target_bin_index == self.current_bin_index: # 제자리 이동 패널티
            reward -= 20

        self.current_bin_index = target_bin_index
        self.bin_fill_levels[target_bin_index] = 0 # 쓰레기통 비움

        # 시간 경과에 따른 쓰레기 증가 및 넘침 패널티
        overflow_penalty = 0
        for i in range(self.num_bins):
            if i != self.current_bin_index:
                new_fill_level = self.bin_fill_levels[i] + self.population_density[i] * time_passed
                if new_fill_level >= 100:
                    overflow_penalty += self.overflow_penalty_weight
                self.bin_fill_levels[i] = new_fill_level

        reward += overflow_penalty
        self.bin_fill_levels = np.clip(self.bin_fill_levels, 0, 100)
        self.steps_taken += 1

        # 종료 조건 확인
        done_success = np.all(self.bin_fill_levels <= self.success_threshold)
        if done_success:
            reward += self.success_bonus # 목표 달성 보너스

        done_timeout = self.steps_taken >= self.max_steps_per_episode
        done = done_success or done_timeout

        next_state = np.concatenate(([self.current_bin_index], self.bin_fill_levels / 100.0))
        return next_state, reward, done

## 2. DQN 모델
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.network(x)

## 3. DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size, self.action_size, self.device = state_size, action_size, device
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99       # 할인율
        self.epsilon = 1.0      # 탐험 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 10 # 타겟 네트워크 업데이트 주기

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # 경험 리플레이 버퍼에 저장
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy 정책으로 액션 선택
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return np.argmax(self.policy_net(state).cpu().data.numpy())

    def learn(self):
        # 경험 리플레이를 통한 학습
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            next_q_values[dones] = 0.0

        target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        # 타겟 네트워크 가중치를 정책 네트워크 가중치로 복사
        self.target_net.load_state_dict(self.policy_net.state_dict())

## 4. 결과 시각화
def plot_results(episode_rewards, learned_path, bin_coordinates, initial_fill_levels):
    plt.figure(figsize=(14, 6))

    # 에피소드별 보상 그래프
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # 학습된 경로 시각화
    plt.subplot(1, 2, 2)
    sizes = initial_fill_levels * 3
    plt.scatter(bin_coordinates[:, 0], bin_coordinates[:, 1], s=sizes, c='red', label='Bins (size ~ fill level)')
    for i, (x, y) in enumerate(bin_coordinates):
        plt.text(x, y, f' {i}({initial_fill_levels[i]:.0f}%)')

    if learned_path:
        path_coords = bin_coordinates[learned_path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], 'b-o', label='Learned Path')

    plt.title("Optimal Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

## 5. 메인 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = WasteCollectionEnv()
    agent = DQNAgent(env.state_space_size, env.action_space_size, device)

    num_episodes = 1000
    episode_rewards = []

    # 학습 루프
    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.learn()

        if e % agent.update_target_every == 0:
            agent.update_target_net()

        episode_rewards.append(total_reward)
        if (e+1) % 50 == 0:
            print(f"Episode: {e+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("Training finished.")

    # 학습된 정책 테스트
    state = env.reset()
    initial_fill_levels = state[1:] * 100
    print("\n--- Testing Learned Policy ---")
    print(f"Initial bin fill levels (%): {np.round(initial_fill_levels, 2)}")

    learned_path = [env.current_bin_index]
    done = False
    agent.epsilon = 0 # 탐험 비활성화

    while not done:
        action = agent.act(state)
        next_state, _, done = env.step(action)
        state = next_state
        learned_path.append(env.current_bin_index)

    print(f"Learned path: {learned_path}")

    # 결과 시각화
    plot_results(episode_rewards, learned_path, env.bin_coordinates, initial_fill_levels)
