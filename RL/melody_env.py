import numpy as np

from music_event import MusicEvent


MOOD_LABELS = [
    "happy",
    "sad",
    "angry",
    "neutral",
    "excited",
    "dark",
    "unstable",
]


MODE_PROFILES = {
    "happy": {
        "contour": "up",
        "preferred_durations": [0.25, 0.5],
        "avoid_durations": [1.5],
        "preferred_velocities": [88, 104],
        "avoid_velocities": [48],
        "tempo": 132,
        "instrument": "Bright Acoustic Piano",
        "chord_progression": ["C", "G", "Am", "F"],
        "range": "mid_high",
        "cadence": "stable",
        "motion": "stepwise",
        "tension": "low",
    },
    "sad": {
        "contour": "down",
        "preferred_durations": [1.0, 1.5],
        "avoid_durations": [0.25],
        "preferred_velocities": [48, 64],
        "avoid_velocities": [120],
        "tempo": 72,
        "instrument": "Acoustic Grand Piano",
        "chord_progression": ["Am", "F", "C", "G"],
        "range": "low_mid",
        "cadence": "stable",
        "motion": "stepwise",
        "tension": "low",
    },
    "angry": {
        "contour": "forceful",
        "preferred_durations": [0.25, 0.5],
        "avoid_durations": [],
        "preferred_velocities": [104, 120],
        "avoid_velocities": [48],
        "tempo": 150,
        "instrument": "Rock Organ",
        "chord_progression": ["Dm", "Bb", "C", "A"],
        "range": "wide",
        "cadence": "strong",
        "motion": "leaps",
        "tension": "medium",
    },
    "neutral": {
        "contour": "balanced",
        "preferred_durations": [0.5, 1.0],
        "avoid_durations": [],
        "preferred_velocities": [64, 88],
        "avoid_velocities": [],
        "tempo": 104,
        "instrument": "Acoustic Grand Piano",
        "chord_progression": ["C", "Am", "F", "G"],
        "range": "mid",
        "cadence": "stable",
        "motion": "stepwise",
        "tension": "low",
    },
    "excited": {
        "contour": "up_leap",
        "preferred_durations": [0.25, 0.5, 1.0],
        "avoid_durations": [1.5],
        "preferred_velocities": [104, 120],
        "avoid_velocities": [48, 64],
        "tempo": 168,
        "instrument": "Electric Piano 1",
        "chord_progression": ["C", "F", "G", "C"],
        "range": "high",
        "cadence": "bright",
        "motion": "leaps",
        "tension": "medium",
    },
    "dark": {
        "contour": "down",
        "preferred_durations": [1.0, 1.5],
        "avoid_durations": [0.25],
        "preferred_velocities": [48, 64],
        "avoid_velocities": [120],
        "tempo": 64,
        "instrument": "Acoustic Grand Piano",
        "chord_progression": ["Am", "Dm", "E", "Am"],
        "range": "low",
        "cadence": "stable",
        "motion": "stepwise",
        "tension": "medium",
    },
    "unstable": {
        "contour": "jagged",
        "preferred_durations": [0.25, 0.5, 1.0],
        "avoid_durations": [],
        "preferred_velocities": [48, 88, 120],
        "avoid_velocities": [],
        "tempo": 124,
        "instrument": "Clavinet",
        "chord_progression": ["Cdim", "F#dim", "Am", "E"],
        "range": "wide",
        "cadence": "tense",
        "motion": "mixed",
        "tension": "high",
    },
}


DEFAULT_REWARD_WEIGHTS = {
    "interval": 1.0,
    "repetition": 1.0,
    "short_loop": 1.0,
    "octave_jump": 1.0,
    "stable_note": 1.0,
    "bar_end": 1.0,
    "pattern": 1.0,
    "diversity": 1.0,
    "duration": 1.0,
    "duration_balance": 1.0,
    "velocity": 1.0,
    "harmony": 1.0,
    "range_profile": 1.0,
    "mode": 1.0,
    "final": 1.0,
}


MOOD_REWARD_WEIGHTS = {
    "happy": {
        "harmony": 1.15,
        "mode": 1.10,
        "duration_balance": 1.15,
        "repetition": 1.15,
        "diversity": 1.10,
    },
    "sad": {
        "duration": 1.25,
        "range_profile": 1.15,
        "mode": 1.15,
        "velocity": 1.10,
        "repetition": 0.90,
        "duration_balance": 0.85,
    },
    "angry": {
        "velocity": 1.20,
        "mode": 1.20,
        "range_profile": 1.10,
        "diversity": 1.10,
        "stable_note": 0.90,
    },
    "neutral": {
        "interval": 1.15,
        "duration": 1.10,
        "harmony": 1.10,
        "mode": 1.10,
    },
    "excited": {
        "repetition": 1.45,
        "short_loop": 1.35,
        "pattern": 1.25,
        "diversity": 1.40,
        "duration_balance": 1.45,
        "velocity": 1.15,
        "mode": 1.15,
        "stable_note": 0.80,
        "bar_end": 0.90,
    },
    "dark": {
        "duration": 1.30,
        "range_profile": 1.25,
        "mode": 1.20,
        "velocity": 1.10,
        "harmony": 1.10,
        "repetition": 0.90,
    },
    "unstable": {
        "mode": 1.25,
        "range_profile": 1.20,
        "velocity": 1.15,
        "diversity": 1.10,
        "stable_note": 0.75,
        "bar_end": 0.75,
        "harmony": 0.90,
    },
}


class MelodyEnv:
    """감정 mode에 따라 reward를 다르게 부여하는 멜로디 생성 환경입니다.

    역할:
    1. action index를 실제 MIDI pitch로 변환합니다.
    2. 현재까지 생성된 melody를 저장합니다.
    3. 선택된 action에 대한 reward를 계산합니다.
    4. 다음 state와 episode 종료 여부를 반환합니다.
    """

    def __init__(
        self,
        mode="neutral",
        melody_length=32,
        state_mode="table",
        mood_vector=None,
        history_size=None,
        durations=None,
        velocities=None,
        octave_expansion=False,
        expansion_start_ratio=0.5,
        max_pitch_jump=12,
        action_masking=True,
        reward_weight_overrides=None,
    ):
        self.mode = mode #감정모드 (happy | sad | angry | neutral | excited | dark | unstable)
        self.mode_profile = MODE_PROFILES.get(mode, MODE_PROFILES["neutral"])
        self.reward_weights = self._build_reward_weights(mode, reward_weight_overrides)
        self.melody_length = melody_length #멜로디 길이 (예를 들어 melody_length가 16이면 모델이 음 16개를 만들었을 때 한 episode가 끝남)
        self.state_mode = state_mode # table: Q-learning용 tuple | vector: DQN용 numpy array
        self.mood_vector = self._normalize_mood_vector(mood_vector)
        self.history_size = history_size if history_size is not None else melody_length
        self.base_notes = self._select_scale(mode) #감정 모드에 따라 사용할 기본 음계를 선택
        self.octave_expansion = octave_expansion
        self.expansion_start_ratio = expansion_start_ratio
        self.max_pitch_jump = max_pitch_jump
        self.action_masking = action_masking
        self.notes = self._build_note_pool(self.base_notes) #실제 action이 선택할 수 있는 음역
        self.base_pitch_actions = {
            index for index, note in enumerate(self.notes) if note in self.base_notes
        }
        self.durations = durations if durations is not None else [0.25, 0.5, 1.0, 1.5]
        self.velocities = velocities if velocities is not None else [48, 64, 88, 104, 120]
        self.pitch_action_size = len(self.notes)
        self.duration_action_size = len(self.durations)
        self.velocity_action_size = len(self.velocities)
        self.rhythm_action_size = self.duration_action_size * self.velocity_action_size
        self.action_size = self.pitch_action_size * self.rhythm_action_size
        self.reset() #처음 상태 초기화 (멜로디 비우기 | step_count 0으로 만들기)

    def _normalize_mood_vector(self, mood_vector):
        """분위기 classifier score를 DQN state에 넣기 좋은 고정 길이 벡터로 만듭니다."""
        if mood_vector is None:
            scores = {label: 0.0 for label in MOOD_LABELS}
            scores[self.mode if self.mode in scores else "neutral"] = 1.0
            return np.array([scores[label] for label in MOOD_LABELS], dtype=np.float32)

        if isinstance(mood_vector, dict):
            values = [float(mood_vector.get(label, 0.0)) for label in MOOD_LABELS]
        else:
            values = [float(value) for value in mood_vector]
            if len(values) != len(MOOD_LABELS):
                raise ValueError(f"mood_vector must have {len(MOOD_LABELS)} values")

        total = sum(values)
        if total <= 0:
            values = [1.0 / len(MOOD_LABELS)] * len(MOOD_LABELS)
        else:
            values = [value / total for value in values]

        return np.array(values, dtype=np.float32)

    def _build_reward_weights(self, mode, reward_weight_overrides=None):
        """Mood별 실패 양상이 다르기 때문에 reward component 가중치를 분리합니다."""
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        weights.update(MOOD_REWARD_WEIGHTS.get(mode, {}))
        if reward_weight_overrides:
            weights.update({
                component: max(0.05, float(value))
                for component, value in reward_weight_overrides.items()
                if component in weights
            })
        return weights

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

    def _build_note_pool(self, base_notes):
        """기본 음계 또는 상하 1옥타브까지 확장한 음계를 구성합니다."""
        if not self.octave_expansion:
            return base_notes

        expanded_notes = {
            note + octave_shift
            for octave_shift in [-12, 0, 12]
            for note in base_notes
        }
        return sorted(expanded_notes)

    def reset(self):
        """새 episode를 시작합니다."""
        self.actions = []
        self.pitch_actions = []
        self.duration_actions = []
        self.velocity_actions = []
        self.events = []
        self.melody = []
        self.note_durations = []
        self.note_velocities = []
        self.reward_breakdowns = []
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

        pitch_action, duration_action, velocity_action = self._decode_action(action)
        note = self.notes[pitch_action] #action을 실제 음으로 변환
        duration = self.durations[duration_action]
        velocity = self.velocities[velocity_action]
        event = MusicEvent(
            tempo=self.mode_profile["tempo"],
            position=self.step_count % 4,
            pitch=note,
            duration=duration,
            velocity=velocity,
        )
        self.actions.append(action) #action인덱스 저장
        self.pitch_actions.append(pitch_action)
        self.duration_actions.append(duration_action)
        self.velocity_actions.append(velocity_action)
        self.events.append(event)
        self.melody.append(note) #실제 note값 저장
        self.note_durations.append(duration)
        self.note_velocities.append(velocity)

        reward_breakdown = self._calculate_reward_breakdown(action)
        reward = sum(reward_breakdown.values()) #방금 고른 음에 대한 reward계산
        self.reward_breakdowns.append(reward_breakdown)

        self.step_count += 1 #음 하나 만들 때마다 count증가 (episode끝나기 직전에 max됨)
        done = self.step_count >= self.melody_length #episode 종료 여부 판단
        next_state = self._get_state() #다음 state계산

        info = {
            "mode": self.mode,
            "melody": self.melody.copy(),
            "actions": self.actions.copy(),
            "pitch_actions": self.pitch_actions.copy(),
            "duration_actions": self.duration_actions.copy(),
            "velocity_actions": self.velocity_actions.copy(),
            "durations": self.note_durations.copy(),
            "velocities": self.note_velocities.copy(),
            "events": [event.to_dict() for event in self.events],
            "reward_breakdowns": [breakdown.copy() for breakdown in self.reward_breakdowns],
        }

        return next_state, reward, done, info

    def get_valid_actions(self):
        """현재 단계에서 agent가 선택해도 되는 action 목록을 반환합니다.

        초반에는 기본 음역만 사용하고, 일정 구간 이후에 상하 1옥타브 확장 음역을 엽니다.
        직전 음에서 너무 멀리 튀는 action은 masking해서 탐색 품질을 안정화합니다.
        """
        if not self.action_masking:
            return list(range(self.action_size))

        candidate_pitch_actions = self._get_valid_pitch_actions()
        valid_actions = []

        for pitch_action in candidate_pitch_actions:
            for duration_action in range(self.duration_action_size):
                for velocity_action in range(self.velocity_action_size):
                    valid_actions.append(self._encode_action(pitch_action, duration_action, velocity_action))

        if valid_actions:
            return valid_actions

        return list(range(self.action_size))

    def _get_valid_pitch_actions(self):
        expansion_start_step = int(self.melody_length * self.expansion_start_ratio)
        expansion_is_open = self.octave_expansion and self.step_count >= expansion_start_step

        if expansion_is_open:
            candidate_pitch_actions = list(range(self.pitch_action_size))
        else:
            candidate_pitch_actions = sorted(self.base_pitch_actions)

        if self.max_pitch_jump is None or not self.melody:
            return candidate_pitch_actions

        previous_note = self.melody[-1]
        masked_pitch_actions = [
            pitch_action for pitch_action in candidate_pitch_actions
            if abs(self.notes[pitch_action] - previous_note) <= self.max_pitch_jump
        ]

        return masked_pitch_actions if masked_pitch_actions else candidate_pitch_actions

    def _decode_action(self, action):
        pitch_action = action // self.rhythm_action_size
        rhythm_action = action % self.rhythm_action_size
        duration_action = rhythm_action // self.velocity_action_size
        velocity_action = rhythm_action % self.velocity_action_size
        return pitch_action, duration_action, velocity_action

    def _encode_action(self, pitch_action, duration_action, velocity_action):
        rhythm_action = duration_action * self.velocity_action_size + velocity_action
        return pitch_action * self.rhythm_action_size + rhythm_action

    def _get_state(self):
        """현재 state를 반환합니다."""
        if self.state_mode == "table":
            return self._get_table_state()
        if self.state_mode == "vector":
            return self._get_state_vector()
        raise ValueError(f"Unknown state_mode: {self.state_mode}")

    def _get_table_state(self):
        """Q-learning baseline용 tuple state를 반환합니다.

        state = (직전 음의 action index, 현재 생성 위치)
        아직 아무 음도 생성하지 않았다면 직전 action은 -1입니다.
        """
        last_action = self.actions[-1] if self.actions else -1
        return (last_action, self.step_count)

    def _get_state_vector(self):
        """DQN용 full padded history state vector를 반환합니다.

        구성:
        - progress: 현재 episode 진행률
        - bar_position: 4박 기준 현재 위치
        - full_history: 지금까지 생성한 전체 action history를 padding한 벡터
        - mood_vector: 분위기 classifier의 score vector
        """
        progress = self.step_count / max(1, self.melody_length)
        bar_position = (self.step_count % 4) / 3.0
        full_history = self._get_full_history_vector()

        base_state = np.array(
            [progress, bar_position],
            dtype=np.float32,
        )

        return np.concatenate([base_state, full_history, self.mood_vector]).astype(np.float32)

    def _get_full_history_vector(self):
        padded_actions = [-1] * max(0, self.history_size - len(self.actions))
        history_actions = self.actions[-self.history_size:] + padded_actions
        return np.array(
            [(action + 1) / self.action_size for action in history_actions],
            dtype=np.float32,
        )

    def get_state_size(self):
        """DQN network input size 계산에 사용할 state 차원을 반환합니다."""
        if self.state_mode == "table":
            return 2
        if self.state_mode == "vector":
            return 2 + self.history_size + len(MOOD_LABELS)
        raise ValueError(f"Unknown state_mode: {self.state_mode}")

    def _calculate_reward(self, action):
        """여러 음악적 기준의 reward를 합산합니다."""
        return sum(self._calculate_reward_breakdown(action).values())

    def _calculate_reward_breakdown(self, action):
        """Reward component를 따로 기록해 어떤 기준이 학습을 이끄는지 확인합니다."""
        raw_breakdown = {
            "interval": self._interval_reward(action),
            "repetition": self._repetition_reward(),
            "short_loop": self._short_loop_penalty(),
            "octave_jump": self._octave_jump_penalty(),
            "stable_note": self._stable_note_reward(action),
            "bar_end": self._bar_end_reward(action),
            "pattern": self._pattern_reward(),
            "diversity": self._diversity_reward(),
            "duration": self._duration_reward(),
            "duration_balance": self._duration_balance_reward(),
            "velocity": self._velocity_reward(),
            "harmony": self._harmony_reward(),
            "range_profile": self._range_profile_reward(),
            "mode": self._mode_reward(action),
            "final": self._final_reward(action),
        }
        return {
            component: value * self.reward_weights.get(component, 1.0)
            for component, value in raw_breakdown.items()
        }

    def _interval_reward(self, action):
        """직전 음과 현재 음 사이의 거리로 자연스러움을 평가합니다."""
        if len(self.actions) < 2:
            return 0.0

        interval = abs(self.melody[-1] - self.melody[-2])

        if interval == 0:
            return -1.0
        if interval <= 2:
            return 2.0
        if interval <= 4:
            return 1.5
        if interval <= 7:
            return 0.5
        return -1.5

    def _repetition_reward(self):
        """같은 음의 연속 반복이 길어질수록 더 강하게 막습니다."""
        if len(self.actions) < 2:
            return 0.0

        last_action = self.pitch_actions[-1]
        repeat_count = 1

        for previous_action in reversed(self.pitch_actions[:-1]):
            if previous_action != last_action:
                break
            repeat_count += 1

        if repeat_count == 2:
            return -2.5
        if repeat_count == 3:
            return -7.0
        if repeat_count >= 4:
            return -13.0

        return 0.0

    def _short_loop_penalty(self):
        """ABAB, ABABAB처럼 짧은 2음 패턴이 과도하게 반복되는 꼼수를 막습니다."""
        if len(self.actions) < 4:
            return 0.0

        # 예: [6, 7, 6, 7]처럼 최근 4개가 ABAB 구조이면 감점합니다.
        pitch_actions = self.pitch_actions

        if pitch_actions[-4] == pitch_actions[-2] and pitch_actions[-3] == pitch_actions[-1]:
            penalty = -3.0

            # 예: [6, 7, 6, 7, 6, 7]처럼 2음 패턴이 3번 반복되면 더 강하게 감점합니다.
            if len(pitch_actions) >= 6:
                if (
                    pitch_actions[-6] == pitch_actions[-4] == pitch_actions[-2]
                    and pitch_actions[-5] == pitch_actions[-3] == pitch_actions[-1]
                ):
                    penalty -= 5.0

            return penalty

        return 0.0

    def _octave_jump_penalty(self):
        """옥타브 단위의 큰 도약은 가능하되 남용하면 감점합니다."""
        if len(self.melody) < 2:
            return 0.0

        interval = abs(self.melody[-1] - self.melody[-2])
        if interval == 12:
            return -2.0
        if interval > 12:
            return -6.0

        return 0.0

    def _stable_note_reward(self, action):
        """각 음계에서 1도, 3도, 5도, 옥타브 음을 안정음으로 봅니다."""
        note = self.melody[-1]
        stable_pitch_classes = {self.base_notes[index] % 12 for index in [0, 2, 4, 7]}
        return 0.7 if note % 12 in stable_pitch_classes else 0.0

    def _bar_end_reward(self, action):
        """4개 음마다 마디 끝이라고 보고 안정적으로 끝나면 보상합니다."""
        note = self.melody[-1]
        position = self.step_count + 1
        stable_pitch_classes = {self.base_notes[index] % 12 for index in [0, 2, 4, 7]}

        if position % 4 == 0:
            return 3.0 if note % 12 in stable_pitch_classes else -1.0

        return 0.0

    def _pattern_reward(self):
        """4음 motif 반복은 아주 약하게 보상하고, 같은 motif 남용은 강하게 감점합니다."""
        if len(self.actions) < 8:
            return 0.0

        last_four = self.pitch_actions[-4:]
        previous_four = self.pitch_actions[-8:-4]

        if last_four != previous_four:
            return self._recent_ngram_penalty(window_size=4)

        if len(set(last_four)) < 3:
            return -3.0

        pattern = tuple(last_four)
        pattern_count = 0
        for start in range(0, len(self.pitch_actions) - 3):
            if tuple(self.pitch_actions[start:start + 4]) == pattern:
                pattern_count += 1

        if pattern_count == 2:
            return 1.0

        return -4.0 * (pattern_count - 2)

    def _recent_ngram_penalty(self, window_size):
        """최근 n-gram이 이미 여러 번 나온 경우 짧은 loop로 보고 감점합니다."""
        if len(self.pitch_actions) < window_size * 2:
            return 0.0

        recent_pattern = tuple(self.pitch_actions[-window_size:])
        pattern_count = 0
        for start in range(0, len(self.pitch_actions) - window_size + 1):
            if tuple(self.pitch_actions[start:start + window_size]) == recent_pattern:
                pattern_count += 1

        if pattern_count <= 1:
            return 0.0

        return -2.0 * (pattern_count - 1)

    def _diversity_reward(self):
        """마지막에 사용 음 다양성을 점검해 단조로운 꼼수를 줄입니다."""
        position = self.step_count + 1
        if position != self.melody_length:
            return 0.0

        reward = 0.0
        unique_ratio = len(set(self.pitch_actions)) / self.pitch_action_size
        same_adjacent_count = sum(
            1 for index in range(1, len(self.pitch_actions))
            if self.pitch_actions[index] == self.pitch_actions[index - 1]
        )
        same_adjacent_ratio = same_adjacent_count / max(1, len(self.pitch_actions) - 1)
        max_two_gram_count = self._max_ngram_count(window_size=2)
        max_four_gram_count = self._max_ngram_count(window_size=4)

        if unique_ratio < 0.45:
            reward -= 28.0
        elif unique_ratio < 0.6:
            reward -= 12.0
        elif unique_ratio >= 0.75:
            reward += 3.0

        if same_adjacent_ratio > 0.15:
            reward -= 36.0 * (same_adjacent_ratio - 0.15)

        if max_two_gram_count >= 5:
            reward -= 4.0 * (max_two_gram_count - 4)

        if max_four_gram_count >= 3:
            reward -= 6.0 * (max_four_gram_count - 2)

        return reward

    def _max_ngram_count(self, window_size):
        if len(self.pitch_actions) < window_size:
            return 0

        counts = {}
        for start in range(0, len(self.pitch_actions) - window_size + 1):
            pattern = tuple(self.pitch_actions[start:start + window_size])
            counts[pattern] = counts.get(pattern, 0) + 1

        return max(counts.values()) if counts else 0

    def _mode_reward(self, action):
        """감정 mode별 특징을 reward에 반영합니다."""
        if len(self.actions) < 2:
            return 0.0

        current_note = self.melody[-1]
        previous_note = self.melody[-2]
        interval = abs(current_note - previous_note)
        contour = self.mode_profile["contour"]
        motion = self.mode_profile["motion"]
        tension = self.mode_profile["tension"]
        reward = 0.0

        if contour == "up":
            reward += 0.7 if current_note > previous_note else -0.1
        elif contour == "down":
            reward += 0.7 if current_note < previous_note else -0.1
        elif contour == "up_leap":
            if current_note > previous_note and interval >= 4:
                reward += 1.0
            elif current_note > previous_note:
                reward += 0.4
        elif contour == "forceful":
            if interval >= 5:
                reward += 1.0
            if interval == 0:
                reward -= 0.8
        elif contour == "balanced":
            reward += 0.5 if 1 <= interval <= 4 else -0.4
        elif contour == "jagged":
            if interval in [1, 6] or interval >= 5:
                reward += 0.7

        if motion == "stepwise" and 1 <= interval <= 4:
            reward += 0.3
        elif motion == "leaps" and interval >= 5:
            reward += 0.4
        elif motion == "mixed" and 1 <= interval <= 7:
            reward += 0.2

        if tension == "high":
            unstable_pitch_classes = {self.base_notes[index] % 12 for index in [1, 3, 5, 6]}
            if current_note % 12 in unstable_pitch_classes:
                reward += 0.6
        elif tension == "low" and interval > 7:
            reward -= 0.4

        return reward

    def _duration_reward(self):
        """감정 mode에 어울리는 음 길이를 약하게 유도합니다."""
        if not self.note_durations:
            return 0.0

        duration = self.note_durations[-1]
        preferred_durations = self.mode_profile["preferred_durations"]
        avoid_durations = self.mode_profile["avoid_durations"]

        if duration in preferred_durations:
            return 0.45
        if duration in avoid_durations:
            return -0.35

        return 0.0

    def _duration_balance_reward(self):
        """빠른 곡도 전부 짧은 음만 쓰지 않도록 마지막에 리듬 균형을 점검합니다."""
        position = self.step_count + 1
        if position != self.melody_length or not self.note_durations:
            return 0.0

        short_note_count = sum(1 for duration in self.note_durations if duration <= 0.5)
        medium_note_count = sum(1 for duration in self.note_durations if duration == 1.0)
        long_note_count = sum(1 for duration in self.note_durations if duration >= 1.5)
        short_note_ratio = short_note_count / len(self.note_durations)
        reward = 0.0

        if self.mode in ["happy", "excited", "angry"]:
            if short_note_ratio > 0.9:
                reward -= 10.0
            elif short_note_ratio > 0.8:
                reward -= 5.0

            if medium_note_count == 0:
                reward -= 4.0
            elif medium_note_count >= 2:
                reward += 2.0

        if self.mode in ["sad", "dark"] and long_note_count == 0:
            reward -= 3.0

        return reward

    def _velocity_reward(self):
        """감정 mode에 어울리는 세기(velocity)를 유도합니다."""
        if not self.note_velocities:
            return 0.0

        velocity = self.note_velocities[-1]
        preferred_velocities = self.mode_profile["preferred_velocities"]
        avoid_velocities = self.mode_profile["avoid_velocities"]

        if velocity in preferred_velocities:
            return 0.45
        if velocity in avoid_velocities:
            return -0.35

        if self.mode_profile["tension"] == "high" and len(self.note_velocities) >= 2:
            previous_velocity = self.note_velocities[-2]
            return 0.25 if abs(velocity - previous_velocity) >= 32 else 0.0

        return 0.0

    def _range_profile_reward(self):
        """Mode profile이 원하는 음역대를 약하게 유도합니다."""
        if not self.melody:
            return 0.0

        note = self.melody[-1]
        low_anchor = self.base_notes[2]
        mid_anchor = self.base_notes[4]
        high_anchor = self.base_notes[5]
        range_profile = self.mode_profile["range"]

        if range_profile == "low":
            return 0.45 if note <= low_anchor else -0.15
        if range_profile == "low_mid":
            return 0.35 if note <= mid_anchor else -0.1
        if range_profile == "mid":
            return 0.3 if low_anchor <= note <= high_anchor else -0.1
        if range_profile == "mid_high":
            return 0.35 if note >= mid_anchor else -0.1
        if range_profile == "high":
            return 0.45 if note >= high_anchor else -0.15
        if range_profile == "wide":
            if len(self.melody) == self.melody_length:
                pitch_range = max(self.melody) - min(self.melody)
                return 1.5 if pitch_range >= 10 else -1.0
        return 0.0

    def _harmony_reward(self):
        """현재 chord template과 melody note가 잘 맞는지 평가합니다."""
        if not self.melody:
            return 0.0

        note_pitch_class = self.melody[-1] % 12
        chord_pitch_classes = self._current_chord_pitch_classes()
        if not chord_pitch_classes:
            return 0.0

        position = self.step_count + 1
        is_bar_end = position % 4 == 0

        if note_pitch_class in chord_pitch_classes:
            return 1.2 if is_bar_end else 0.55

        scale_pitch_classes = {note % 12 for note in self.base_notes}
        if note_pitch_class in scale_pitch_classes:
            return -0.15

        return -0.8

    def _current_chord_pitch_classes(self):
        chord_progression = self.mode_profile.get("chord_progression", [])
        if not chord_progression:
            return set()

        chord_index = (self.step_count // 4) % len(chord_progression)
        chord_name = chord_progression[chord_index]
        return self._chord_to_pitch_classes(chord_name)

    def _chord_to_pitch_classes(self, chord_name):
        root_name = chord_name
        quality = "major"

        if chord_name.endswith("dim"):
            root_name = chord_name[:-3]
            quality = "dim"
        elif chord_name.endswith("m"):
            root_name = chord_name[:-1]
            quality = "minor"

        root_pitch_classes = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
        }
        root = root_pitch_classes.get(root_name)
        if root is None:
            return set()

        intervals_by_quality = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "dim": [0, 3, 6],
        }
        return {
            (root + interval) % 12
            for interval in intervals_by_quality[quality]
        }

    def _final_reward(self, action):
        """마지막 음이 mode에 맞게 끝나도록 보상합니다."""
        note = self.melody[-1]
        position = self.step_count + 1
        if position != self.melody_length:
            return 0.0

        if self.mode in ["happy", "neutral", "excited"]:
            if note % 12 == self.base_notes[0] % 12:
                return 10.0
            if note % 12 in {self.base_notes[2] % 12, self.base_notes[4] % 12}:
                return 5.0
            return -3.0

        if self.mode in ["sad", "dark"]:
            if note % 12 == self.base_notes[0] % 12:
                return 10.0
            if note % 12 in {self.base_notes[2] % 12, self.base_notes[4] % 12}:
                return 4.0
            return -3.0

        if self.mode == "angry":
            if note % 12 in {
                self.base_notes[0] % 12,
                self.base_notes[4] % 12,
                self.base_notes[7] % 12,
            }:
                return 6.0
            return -1.0

        if self.mode == "unstable":
            unstable_pitch_classes = {self.base_notes[index] % 12 for index in [1, 3, 5, 6]}
            if note % 12 in unstable_pitch_classes:
                return 4.0
            return 0.0

        return 0.0
