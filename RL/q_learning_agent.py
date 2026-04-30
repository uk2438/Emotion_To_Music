import random
from collections import defaultdict

import numpy as np

#Q-table구조 : q_table[state] = [action0_score, action1_score, action2_score, ...]

class QLearningAgent:
    """Q-learning 기반 멜로디 생성 에이전트입니다."""
    #Agent가 action선택 -> Environment가 reward줌 -> Agent가 Q_table 업데이트
    #Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]

    def __init__(
        self,
        action_size, #행동 개수 (예를 들어 C major라고 치면 '도(0) 레(1) 미(2) 파(3) 솔(4) 라(5) 시(6) 도(7)'이므로 총 8개의 action존재)
        alpha=0.1, #Learning Rate
        gamma=0.95, #discount factor (높게 설정한 이유는 현재 음 하나만 중요한게 아니라 이 음이 나중에 좋은 멜로디로 이어지는게 더 중요하기 때문이에요 -> 나중에 수정 가능)
        epsilon=1.0, #초기 epsilon value : 초기에는 아무것도 모르니까 랜덤하게 많이 골라야됌. 그래서 100% exploration할 수 있도록 했어요
        epsilon_decay=0.995, #한 episode가 끝날 때마다 epsilon을 조금씩 줄여나가야됨 -> optimal policy를 얻어야 하므로
        epsilon_min=0.05, #epsilon value의 min값. 0이 되버리면 아예 exploration을 안해버리니까 하한선 필요할듯요
    ):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # state별 action value를 저장합니다.
        # 어떤 상태에서 어떤 action이 얼마나 좋은지 저장하는 table
        # 예를 들어서 state가 [직전 음 = 도, 현재 위치 = 3번째 음]이면 state = (0, 3)이며 q_table[(0,3)] = [0.01, 3.2, 1.5, -0.2, 0.8, -1.1, -2.0, 0.4]이면 다음 행동으로는 action1이 가장 좋은 것임
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

    def choose_action(self, state, training=True):
        """epsilon-greedy 방식으로 다음 action을 선택합니다."""
        #random.random()은 0이상 1미만의  랜덤한 숫자를 만들고 이 값이 epsilon보다 작으면 exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        #exploitation할 때
        q_values = self.q_table[state] #현재 Q_table 가지고 오기
        max_q = np.max(q_values)

        # 같은 max Q-value가 여러 개면 랜덤 선택해서 한 음만 과도하게 고르는 현상을 줄여야 해요
        best_actions = np.where(q_values == max_q)[0] #인덱스 가지고 오기
        return int(random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """Q-learning 업데이트 식을 적용합니다."""

        #현재 q값 가지고오기
        current_q = self.q_table[state][action]

        #episode가 끝났다면 (마지막 음을 골라서 끝난 경우) 그냥 끝
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """episode가 진행될수록 exploration 비율을 줄입니다."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)