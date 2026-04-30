class MelodyEnv:
    """감정 mode에 따라 reward를 다르게 부여하는 멜로디 생성 환경입니다.

    역할:
    1. action index를 실제 MIDI pitch로 변환합니다.
    2. 현재까지 생성된 melody를 저장합니다.
    3. 선택된 action에 대한 reward를 계산합니다.
    4. 다음 state와 episode 종료 여부를 반환합니다.
    """

    def __init__(self, mode="neutral", melody_length=32):
        self.mode = mode #감정모드 (happy | sad | angry | neutral | excited | dark | unstable)
        self.melody_length = melody_length #멜로디 길이 (예를 들어 melody_length가 16이면 모델이 음 16개를 만들었을 때 한 episode가 끝남)
        self.notes = self._select_scale(mode) #감정 모드에 따라 사용할 음계를 선택
        self.action_size = len(self.notes) #사용 가능한 음 개수
        self.reset() #처음 상태 초기화 (멜로디 비우기 | step_count 0으로 만들기)

    def _select_scale(self, mode):
        """감정 mode에 맞는 음계를 선택합니다. 각 숫자는 MIDI pitch입니다."""
        #60 = 도 | 62 = 레 | 64 = 미 | 65 = 파 | 67 = 솔 | 69 = 라 | 71 = 시 | 72 = 높은 도
        scales = {
            # C major: 도 레 미 파 솔 라 시 높은도
            "happy": [60, 62, 64, 65, 67, 69, 71, 72],
            "neutral": [60, 62, 64, 65, 67, 69, 71, 72],

            # A minor: 라 시 도 레 미 파 솔 높은라
            "sad": [57, 59, 60, 62, 64, 65, 67, 69],
            "dark": [57, 59, 60, 62, 64, 65, 67, 69],

            # D minor 느낌: 강한 느낌을 위해 약간 어두운 음계를 사용합니다.
            "angry": [62, 64, 65, 67, 69, 70, 72, 74],

            # excited는 밝고 높은 음역으로 올라가는 느낌을 주기 위해 구성했습니다.
            "excited": [60, 62, 64, 67, 69, 72, 74, 76],

            # unstable은 반음과 불안정한 음을 포함해서 긴장감을 만듭니다.
            "unstable": [60, 61, 64, 66, 67, 70, 71, 72],
        }

        return scales.get(mode, scales["neutral"])

    def reset(self):
        """새 episode를 시작합니다."""
        self.actions = []
        self.melody = []
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        """Agent가 선택한 action을 환경에 반영합니다.

        반환값:
        next_state: 다음 상태
        reward: 현재 action에 대한 보상
        done: melody_length만큼 생성했는지 여부
        info: 디버깅/출력용 부가 정보
        """
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action: {action}")

        note = self.notes[action] #action을 실제 음으로 변환
        self.actions.append(action) #action인덱스 저장
        self.melody.append(note) #실제 note값 저장

        reward = self._calculate_reward(action) #방금 고른 음에 대한 reward계산

        self.step_count += 1 #음 하나 만들 때마다 count증가 (episode끝나기 직전에 max됨)
        done = self.step_count >= self.melody_length #episode 종료 여부 판단
        next_state = self._get_state() #다음 state계산

        info = {
            "mode": self.mode,
            "melody": self.melody.copy(),
            "actions": self.actions.copy(),
        }

        return next_state, reward, done, info

    def _get_state(self):
        """현재 state를 반환합니다.

        state = (직전 음의 action index, 현재 생성 위치)
        아직 아무 음도 생성하지 않았다면 직전 action은 -1입니다.
        """
        last_action = self.actions[-1] if self.actions else -1
        return (last_action, self.step_count)

    ## 이 부분을 같이 수정해야해요..! constraint를 추가할소록 퀄리티 업~
    def _calculate_reward(self, action):
        """여러 음악적 기준의 reward를 합산합니다."""
        reward = 0.0
        reward += self._interval_reward(action)
        reward += self._repetition_reward()
        reward += self._short_loop_penalty()
        reward += self._stable_note_reward(action)
        reward += self._bar_end_reward(action)
        reward += self._pattern_reward()
        reward += self._mode_reward(action)
        reward += self._final_reward(action)
        return reward

    def _interval_reward(self, action):
        """직전 음과 현재 음 사이의 거리로 자연스러움을 평가합니다."""
        if len(self.actions) < 2:
            return 0.0

        prev_action = self.actions[-2]
        interval = abs(action - prev_action)

        if interval == 0:
            return -0.3
        if interval == 1:
            return 2.0
        if interval == 2:
            return 1.5
        if interval == 3:
            return 0.5
        return -1.5

    def _repetition_reward(self):
        """같은 음이 3번 연속 반복되는 것을 강하게 막습니다."""
        if len(self.actions) < 3:
            return 0.0

        if self.actions[-1] == self.actions[-2] == self.actions[-3]:
            return -3.0

        return 0.0

    def _short_loop_penalty(self):
        """ABAB, ABABAB처럼 짧은 2음 패턴이 과도하게 반복되는 꼼수를 막습니다."""
        if len(self.actions) < 4:
            return 0.0

        # 예: [6, 7, 6, 7]처럼 최근 4개가 ABAB 구조이면 감점합니다.
        if self.actions[-4] == self.actions[-2] and self.actions[-3] == self.actions[-1]:
            penalty = -2.0

            # 예: [6, 7, 6, 7, 6, 7]처럼 2음 패턴이 3번 반복되면 더 강하게 감점합니다.
            if len(self.actions) >= 6:
                if (
                    self.actions[-6] == self.actions[-4] == self.actions[-2]
                    and self.actions[-5] == self.actions[-3] == self.actions[-1]
                ):
                    penalty -= 3.0

            return penalty

        return 0.0

    def _stable_note_reward(self, action):
        """각 음계에서 1도, 3도, 5도, 옥타브 음을 안정음으로 봅니다."""
        stable_actions = [0, 2, 4, 7]
        return 0.7 if action in stable_actions else 0.0

    def _bar_end_reward(self, action):
        """4개 음마다 마디 끝이라고 보고 안정적으로 끝나면 보상합니다."""
        position = self.step_count + 1
        stable_actions = [0, 2, 4, 7]

        if position % 4 == 0:
            return 3.0 if action in stable_actions else -1.0

        return 0.0

    def _pattern_reward(self):
        """최근 4개 패턴이 바로 이전 4개 패턴과 같으면 반복 motif로 보고 보상합니다."""
        if len(self.actions) < 8:
            return 0.0

        last_four = self.actions[-4:]
        previous_four = self.actions[-8:-4]

        if last_four == previous_four:
            return 4.0

        return 0.0

    def _mode_reward(self, action):
        """감정 mode별 특징을 reward에 반영합니다."""
        if len(self.actions) < 2:
            return 0.0

        prev_action = self.actions[-2]
        interval = abs(action - prev_action)

        if self.mode == "happy":
            # 행복한 느낌: 상행 진행을 살짝 선호합니다.
            return 0.6 if action > prev_action else 0.0

        if self.mode == "sad":
            # 슬픈 느낌: 하행 진행을 살짝 선호합니다.
            return 0.6 if action < prev_action else 0.0

        if self.mode == "angry":
            # 화남: 큰 도약과 강한 움직임을 약간 선호합니다.
            if interval >= 3:
                return 0.9
            if interval == 0:
                return -0.8
            return 0.0

        if self.mode == "neutral":
            # 중립: 안정적인 작은 움직임을 선호합니다.
            return 0.5 if interval <= 2 else -0.5

        if self.mode == "excited":
            # 놀람/흥분: 위로 도약하는 움직임을 선호합니다.
            if action > prev_action and interval >= 2:
                return 1.0
            return 0.2 if interval >= 2 else 0.0

        if self.mode == "dark":
            # 두려움/어두움: 낮은 음역과 하행 진행을 선호합니다.
            reward = 0.0
            if action < prev_action:
                reward += 0.7
            if action <= 2:
                reward += 0.5
            return reward

        if self.mode == "unstable":
            # 혐오/불안정: 반음, 불안정 음, 큰 움직임을 약간 허용합니다.
            unstable_actions = [1, 3, 5, 6]
            reward = 0.6 if action in unstable_actions else 0.0
            if interval >= 3:
                reward += 0.4
            return reward

        return 0.0

    def _final_reward(self, action):
        """마지막 음이 mode에 맞게 끝나도록 보상합니다."""
        position = self.step_count + 1
        if position != self.melody_length:
            return 0.0

        if self.mode in ["happy", "neutral", "excited"]:
            if action in [0, 7]:
                return 10.0
            if action in [2, 4]:
                return 5.0
            return -3.0

        if self.mode in ["sad", "dark"]:
            if action in [0, 7]:
                return 10.0
            if action in [2, 4]:
                return 4.0
            return -3.0

        if self.mode == "angry":
            if action in [0, 4, 7]:
                return 6.0
            return -1.0

        if self.mode == "unstable":
            if action in [1, 3, 5, 6]:
                return 4.0
            return 0.0

        return 0.0