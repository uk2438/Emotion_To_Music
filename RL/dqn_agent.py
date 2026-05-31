import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """DQN 학습에 사용할 경험 replay buffer입니다."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_valid_actions=None):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
                None if next_valid_actions is None else list(next_valid_actions),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            list(next_valid_actions),
        )

    def __len__(self):
        return len(self.buffer)


class SimpleQNetwork:
    """numpy만 사용하는 작은 2-layer Q-network입니다."""

    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.w1 = np.random.randn(state_size, hidden_size).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(hidden_size, action_size).astype(np.float32) * 0.1
        self.b2 = np.zeros(action_size, dtype=np.float32)

    def predict(self, states):
        states = np.array(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        hidden_pre = states @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        q_values = hidden @ self.w2 + self.b2
        return q_values

    def train_batch(self, states, actions, targets, learning_rate):
        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        batch_size = states.shape[0]

        hidden_pre = states @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        q_values = hidden @ self.w2 + self.b2

        predicted = q_values[np.arange(batch_size), actions]
        errors = predicted - targets
        loss = float(np.mean(errors**2))

        dq = np.zeros_like(q_values)
        dq[np.arange(batch_size), actions] = (2.0 / batch_size) * errors

        dw2 = hidden.T @ dq
        db2 = np.sum(dq, axis=0)
        dhidden = dq @ self.w2.T
        dhidden_pre = dhidden * (hidden_pre > 0)
        dw1 = states.T @ dhidden_pre
        db1 = np.sum(dhidden_pre, axis=0)

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

        return loss

    def copy_from(self, other):
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()

    def get_weights(self):
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
        }

    def set_weights(self, weights):
        self.w1 = weights["w1"].copy()
        self.b1 = weights["b1"].copy()
        self.w2 = weights["w2"].copy()
        self.b2 = weights["b2"].copy()


class DQNAgent:
    """DQN 기반 action 선택 에이전트입니다.

    use_double_dqn=True이면 다음 상태의 action 선택은 online network가 하고,
    그 action의 Q-value 평가는 target network가 수행합니다. 이렇게 분리하면
    큰 action space에서 Q-value가 과대평가되는 문제를 줄일 수 있습니다.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_capacity=10000,
        use_double_dqn=True,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_double_dqn = use_double_dqn

        self.q_network = SimpleQNetwork(state_size, action_size, hidden_size)
        self.target_network = SimpleQNetwork(state_size, action_size, hidden_size)
        self.target_network.copy_from(self.q_network)
        self.memory = ReplayBuffer(capacity=replay_capacity)

    def choose_action(self, state, training=True, valid_actions=None):
        """epsilon-greedy 방식으로 다음 action을 선택합니다."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_network.predict(state)[0]
        valid_q_values = q_values[valid_actions]
        max_q = np.max(valid_q_values)
        best_actions = [action for action in valid_actions if q_values[action] == max_q]
        return int(random.choice(best_actions))

    def remember(self, state, action, reward, next_state, done, next_valid_actions=None):
        self.memory.add(state, action, reward, next_state, done, next_valid_actions)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(batch_size)
        online_next_q_values = self.q_network.predict(next_states)
        target_next_q_values = self.target_network.predict(next_states)
        max_next_q = []
        for index, valid_actions in enumerate(next_valid_actions):
            if valid_actions is None:
                valid_actions = list(range(self.action_size))

            if self.use_double_dqn:
                online_q_values = online_next_q_values[index][valid_actions]
                best_valid_index = int(np.argmax(online_q_values))
                best_action = valid_actions[best_valid_index]
                max_next_q.append(target_next_q_values[index][best_action])
            else:
                max_next_q.append(np.max(target_next_q_values[index][valid_actions]))

        max_next_q = np.array(max_next_q, dtype=np.float32)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        return self.q_network.train_batch(
            states=states,
            actions=actions,
            targets=targets,
            learning_rate=self.learning_rate,
        )

    def update_target_network(self):
        self.target_network.copy_from(self.q_network)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_model_state(self):
        return {
            "q_network": self.q_network.get_weights(),
            "target_network": self.target_network.get_weights(),
        }

    def load_model_state(self, model_state):
        self.q_network.set_weights(model_state["q_network"])
        self.target_network.set_weights(model_state["target_network"])


class FactorizedDQNAgent:
    """pitch, duration, velocity를 분리해서 학습하는 DQN agent입니다.

    하나의 network가 pitch-duration-velocity 조합 전체를 직접 고르는 대신,
    음악 event를 구성하는 factor별 Q-network를 따로 둡니다. 이렇게 하면
    action space가 커져도 각 음악 variable의 선호를 더 명확하게 학습할 수
    있고, reference repo의 event-token 생성 방식으로 확장하기 쉬워집니다.
    """

    def __init__(
        self,
        state_size,
        pitch_action_size,
        duration_action_size,
        velocity_action_size,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_capacity=10000,
        use_double_dqn=True,
    ):
        self.state_size = state_size
        self.pitch_action_size = pitch_action_size
        self.duration_action_size = duration_action_size
        self.velocity_action_size = velocity_action_size
        self.rhythm_action_size = duration_action_size * velocity_action_size
        self.action_size = pitch_action_size * self.rhythm_action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_double_dqn = use_double_dqn

        self.pitch_network = SimpleQNetwork(state_size, pitch_action_size, hidden_size)
        self.duration_network = SimpleQNetwork(state_size, duration_action_size, hidden_size)
        self.velocity_network = SimpleQNetwork(state_size, velocity_action_size, hidden_size)
        self.target_pitch_network = SimpleQNetwork(state_size, pitch_action_size, hidden_size)
        self.target_duration_network = SimpleQNetwork(state_size, duration_action_size, hidden_size)
        self.target_velocity_network = SimpleQNetwork(state_size, velocity_action_size, hidden_size)
        self.update_target_network()
        self.memory = ReplayBuffer(capacity=replay_capacity)

    def choose_action(self, state, training=True, valid_actions=None):
        """valid action 중 factor별 Q 평균이 가장 큰 action을 선택합니다."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        pitch_q_values = self.pitch_network.predict(state)[0]
        duration_q_values = self.duration_network.predict(state)[0]
        velocity_q_values = self.velocity_network.predict(state)[0]
        best_actions = self._best_combined_actions(
            valid_actions=valid_actions,
            pitch_q_values=pitch_q_values,
            duration_q_values=duration_q_values,
            velocity_q_values=velocity_q_values,
        )
        return int(random.choice(best_actions))

    def remember(self, state, action, reward, next_state, done, next_valid_actions=None):
        self.memory.add(state, action, reward, next_state, done, next_valid_actions)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(batch_size)
        pitch_actions, duration_actions, velocity_actions = self._decode_actions(actions)

        online_next_pitch_q = self.pitch_network.predict(next_states)
        online_next_duration_q = self.duration_network.predict(next_states)
        online_next_velocity_q = self.velocity_network.predict(next_states)
        target_next_pitch_q = self.target_pitch_network.predict(next_states)
        target_next_duration_q = self.target_duration_network.predict(next_states)
        target_next_velocity_q = self.target_velocity_network.predict(next_states)

        max_next_q = []
        for index, valid_actions in enumerate(next_valid_actions):
            if valid_actions is None:
                valid_actions = list(range(self.action_size))

            if self.use_double_dqn:
                best_actions = self._best_combined_actions(
                    valid_actions=valid_actions,
                    pitch_q_values=online_next_pitch_q[index],
                    duration_q_values=online_next_duration_q[index],
                    velocity_q_values=online_next_velocity_q[index],
                )
                best_action = random.choice(best_actions)
                best_pitch_action, best_duration_action, best_velocity_action = self._decode_action(best_action)
                next_value = self._combine_q_values(
                    target_next_pitch_q[index][best_pitch_action],
                    target_next_duration_q[index][best_duration_action],
                    target_next_velocity_q[index][best_velocity_action],
                )
            else:
                combined_values = [
                    self._combined_q_for_action(
                        action,
                        target_next_pitch_q[index],
                        target_next_duration_q[index],
                        target_next_velocity_q[index],
                    )
                    for action in valid_actions
                ]
                next_value = max(combined_values)

            max_next_q.append(next_value)

        max_next_q = np.array(max_next_q, dtype=np.float32)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)
        pitch_loss = self.pitch_network.train_batch(
            states=states,
            actions=pitch_actions,
            targets=targets,
            learning_rate=self.learning_rate,
        )
        duration_loss = self.duration_network.train_batch(
            states=states,
            actions=duration_actions,
            targets=targets,
            learning_rate=self.learning_rate,
        )
        velocity_loss = self.velocity_network.train_batch(
            states=states,
            actions=velocity_actions,
            targets=targets,
            learning_rate=self.learning_rate,
        )

        return (pitch_loss + duration_loss + velocity_loss) / 3.0

    def update_target_network(self):
        self.target_pitch_network.copy_from(self.pitch_network)
        self.target_duration_network.copy_from(self.duration_network)
        self.target_velocity_network.copy_from(self.velocity_network)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_model_state(self):
        return {
            "pitch_network": self.pitch_network.get_weights(),
            "duration_network": self.duration_network.get_weights(),
            "velocity_network": self.velocity_network.get_weights(),
            "target_pitch_network": self.target_pitch_network.get_weights(),
            "target_duration_network": self.target_duration_network.get_weights(),
            "target_velocity_network": self.target_velocity_network.get_weights(),
        }

    def load_model_state(self, model_state):
        self.pitch_network.set_weights(model_state["pitch_network"])
        self.duration_network.set_weights(model_state["duration_network"])
        self.velocity_network.set_weights(model_state["velocity_network"])
        self.target_pitch_network.set_weights(model_state["target_pitch_network"])
        self.target_duration_network.set_weights(model_state["target_duration_network"])
        self.target_velocity_network.set_weights(model_state["target_velocity_network"])

    def _decode_action(self, action):
        pitch_action = int(action) // self.rhythm_action_size
        rhythm_action = int(action) % self.rhythm_action_size
        duration_action = rhythm_action // self.velocity_action_size
        velocity_action = rhythm_action % self.velocity_action_size
        return pitch_action, duration_action, velocity_action

    def _decode_actions(self, actions):
        actions = np.array(actions, dtype=np.int64)
        pitch_actions = actions // self.rhythm_action_size
        rhythm_actions = actions % self.rhythm_action_size
        duration_actions = rhythm_actions // self.velocity_action_size
        velocity_actions = rhythm_actions % self.velocity_action_size
        return pitch_actions, duration_actions, velocity_actions

    def _best_combined_actions(self, valid_actions, pitch_q_values, duration_q_values, velocity_q_values):
        combined_values = [
            self._combined_q_for_action(action, pitch_q_values, duration_q_values, velocity_q_values)
            for action in valid_actions
        ]
        max_value = max(combined_values)
        return [
            action for action, value in zip(valid_actions, combined_values)
            if value == max_value
        ]

    def _combined_q_for_action(self, action, pitch_q_values, duration_q_values, velocity_q_values):
        pitch_action, duration_action, velocity_action = self._decode_action(action)
        return self._combine_q_values(
            pitch_q_values[pitch_action],
            duration_q_values[duration_action],
            velocity_q_values[velocity_action],
        )

    def _combine_q_values(self, pitch_q_value, duration_q_value, velocity_q_value):
        return (
            float(pitch_q_value)
            + float(duration_q_value)
            + float(velocity_q_value)
        ) / 3.0
