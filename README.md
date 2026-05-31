# EmotionRL Music Generation

## 1. Motivation

이 프로젝트의 목표는 **이미지의 감정 분위기를 해석하고, 그 분위기에 맞는 음악을 생성하는 강화학습 기반 음악 생성 시스템**을 만드는 것입니다.

단순히 MIDI 음표를 랜덤하게 생성하는 것이 아니라, 다음 흐름을 하나의 pipeline으로 연결합니다.

```text
image
-> visual emotion classification
-> music mood conditioning
-> reinforcement learning environment
-> factorized DQN music event generation
-> MIDI export
-> user feedback
-> reward/selection improvement
```

즉 입력은 이미지이고, 출력은 감정 분위기에 맞는 MIDI 음악입니다.

이 프로젝트가 다루는 핵심 문제는 다음과 같습니다.

```text
이미지에서 느껴지는 감정 분위기를 어떻게 음악 생성 조건으로 바꿀 것인가?
음악을 강화학습의 state/action/reward 구조로 어떻게 표현할 것인가?
pitch뿐 아니라 duration, velocity, chord, rhythm을 어떻게 함께 다룰 것인가?
사람이 들었을 때 좋은 음악이라는 평가를 어떻게 학습 과정에 반영할 것인가?
```

현재 구현은 우리가 구상한 다음 단계인 Transformer 기반 sequence model과 learned reward 구조로 확장하기 전 단계입니다. 지금은 **image-conditioned factorized DQN + rule-based reward + user feedback reward tuning** 구조를 먼저 안정화하고 있습니다.

## 2. 전체 프로젝트 구조

현재 전체 구조는 다음과 같습니다.

```mermaid
flowchart LR
    IMG["Image Upload"] --> CLS["EfficientNet-B0 Emotion Classifier"]

    CLS --> RAW["Raw EmoSet Output<br/>8-class probabilities"]
    RAW --> MAP["Music Mood Remapping"]

    MAP --> MODE["Top Music Mood Label"]
    MAP --> MV["7-class Mood Vector"]

    MODE --> ENV["MelodyEnv"]
    MV --> ENV
    FBW["Feedback-based<br/>Reward Weight Overrides"] --> ENV

    ENV --> STATE["Vector State<br/>progress + bar position<br/>action history + mood vector"]
    STATE --> AGENT["Factorized DQN Agent"]

    AGENT --> PQ["Pitch Q Network"]
    AGENT --> DQ["Duration Q Network"]
    AGENT --> VQ["Velocity Q Network"]

    PQ --> ACT["Event Action"]
    DQ --> ACT
    VQ --> ACT

    ACT --> STEP["Environment Step"]
    STEP --> NEXT["Next State"]
    STEP --> REWARD["Mood-weighted Rule Reward<br/>melody, rhythm, velocity, harmony"]

    STATE --> BUFFER["Replay Buffer"]
    ACT --> BUFFER
    REWARD --> BUFFER
    NEXT --> BUFFER

    BUFFER --> AGENT

    STEP --> MIDI["MIDI Export<br/>melody + chord pad + bass"]
    STEP --> GEN["Generation Log<br/>events, metrics, reward breakdown"]

    MIDI --> GUI["Feedback GUI"]
    GEN --> GUI

    GUI --> USER["5-point User Feedback"]
    USER --> LOG["feedback_log.jsonl / csv"]
    USER --> SCORE["Feedback-adjusted<br/>Selection Score"]
    USER --> UPDATE["Update Reward<br/>Weight Overrides"]

    UPDATE --> FBW
```

큰 흐름은 네 부분으로 나눌 수 있습니다.

```text
1. 이미지 감정 분류
2. 감정 분류 결과를 음악 생성용 mood로 변환
3. mood-conditioned RL 환경에서 음악 event 생성
4. 사용자 피드백을 저장하고 다음 생성에 반영
```

프로젝트 주요 파일은 다음과 같습니다.

```text
RL/team_emotion_classifier.py  # EfficientNet classifier wrapper + mood remapping
RL/melody_env.py               # 강화학습 환경, mood profile, reward 계산
RL/dqn_agent.py                # DQN / Factorized DQN 구현
RL/main.py                     # 단일 이미지/단일 mood 생성 실행
RL/run_mood_batch.py           # 여러 mood batch 생성 및 CSV 비교
RL/feedback.py                 # 사용자 피드백 저장, 피드백 점수, reward weight 조정
RL/feedback_gui.py             # 로컬 feedback GUI 서버
RL/music_event.py              # tempo, position, pitch, duration, velocity event 표현
```

## 3. EfficientNet Classifier

이미지 감정 분류는 EfficientNet-B0 기반 classifier를 사용합니다.

입력 이미지는 다음 전처리를 거칩니다.

```text
Resize: 224 x 224
ToTensor
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

classifier는 이미지를 보고 EmoSet 기준의 8개 class 확률을 출력합니다.

```text
amusement
anger
awe
contentment
disgust
excitement
fear
sadness
```

모델 출력 logits를 `z = [z_1, z_2, ..., z_8]`라고 하면, 각 EmoSet class 확률은 softmax로 계산됩니다.

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{8}\exp(z_j)}
$$

즉 classifier의 raw output은 다음과 같은 확률 벡터입니다.

$$
\mathbf{p}_{emo}
=
\left[
p_{amusement},
p_{anger},
p_{awe},
p_{contentment},
p_{disgust},
p_{excitement},
p_{fear},
p_{sadness}
\right]
$$

주의할 점은 이 8개 label은 **시각 감정 분류 label**이라는 것입니다. 음악 생성기가 직접 사용하는 label은 아닙니다. 예를 들어 `awe`는 시각적으로는 경외감이지만, 음악적으로는 밝고 웅장할 수도 있고 어둡고 신비로울 수도 있습니다. 그래서 classifier 결과를 바로 RL label로 쓰지 않고, 음악 생성용 mood 공간으로 다시 변환합니다.

## 4. 분류 결과를 음악 생성용 Mood로 변환하는 과정

강화학습 환경이 사용하는 음악 mood는 7개입니다.

```text
happy
neutral
sad
dark
angry
excited
unstable
```

EmoSet 8-class 확률을 다음 변수로 둡니다.

$$
\begin{aligned}
A_m &= p_{amusement} \\
A_g &= p_{anger} \\
A_w &= p_{awe} \\
C &= p_{contentment} \\
D &= p_{disgust} \\
E &= p_{excitement} \\
F &= p_{fear} \\
S &= p_{sadness}
\end{aligned}
$$

이 프로젝트는 raw emotion probability를 바로 하나의 label로 자르지 않고, 여러 감정이 하나의 음악 mood에 영향을 줄 수 있도록 weighted remapping을 사용합니다.

먼저 normalization 전 music mood score를 계산합니다.

$$
\begin{aligned}
u_{happy} &= A_m + 0.4C \\
u_{neutral} &= C + 0.3A_w \\
u_{sad} &= S \\
u_{dark} &= 0.6F + 0.5D + 0.4S + 0.3A_w \\
u_{angry} &= A_g + 0.4D \\
u_{excited} &= E + 0.4A_m + 0.3A_w \\
u_{unstable} &= 0.6F + 0.4A_g + 0.3D + 0.3E
\end{aligned}
$$

이후 7개 score를 합이 1이 되도록 정규화합니다.

$$
q_k =
\frac{u_k}
{u_{happy}+u_{neutral}+u_{sad}+u_{dark}+u_{angry}+u_{excited}+u_{unstable}+\epsilon}
$$

여기서 \(M\)은 7개 music mood의 집합입니다.

$$
M = \{happy, neutral, sad, dark, angry, excited, unstable\}
$$

최종 music mood vector는 다음과 같습니다.

$$
\mathbf{q}_{mood}
=
\left[
q_{happy},
q_{neutral},
q_{sad},
q_{dark},
q_{angry},
q_{excited},
q_{unstable}
\right]
$$

그리고 가장 높은 확률을 가진 mood가 MelodyEnv의 대표 mode가 됩니다.

$$
mode = \arg\max_{k \in M} q_k
$$

예를 들어 classifier가 다음과 같이 판단했다고 해보겠습니다.

```text
amusement: 0.60
excitement: 0.25
contentment: 0.10
awe: 0.05
others: 0.00
```

그러면 주요 score는 다음처럼 계산됩니다.

$$
\begin{aligned}
u_{happy} &= 0.60 + 0.4(0.10) = 0.64 \\
u_{excited} &= 0.25 + 0.4(0.60) + 0.3(0.05) = 0.505 \\
u_{neutral} &= 0.10 + 0.3(0.05) = 0.115
\end{aligned}
$$

이 경우 top mood는 `happy`가 될 가능성이 높지만, `excited` 확률도 mood vector 안에 함께 남습니다. 그래서 DQN state에는 단순 label뿐 아니라 전체 mood probability distribution도 들어갑니다.

## 5. MelodyEnv가 Mood를 보고 음악 환경을 구성하는 과정

`MelodyEnv`는 강화학습에서 환경 역할을 합니다. 이 환경은 mode를 보고 다음 요소들을 선택합니다.

```text
scale
tempo
instrument
chord progression
preferred durations
preferred velocities
pitch range profile
melodic contour
motion style
tension level
reward weights
```

예를 들어 `excited` mode는 다음 profile을 사용합니다.

```text
scale: [60, 62, 64, 67, 69, 72, 74, 76]
tempo: 168
instrument: Electric Piano 1
chord progression: C - F - G - C
preferred durations: 0.25, 0.5, 1.0
avoid durations: 1.5
preferred velocities: 104, 120
avoid velocities: 48, 64
range: high
contour: up_leap
motion: leaps
tension: medium
```

이 말은 `excited`일 때 환경이 다음과 같은 음악을 더 선호한다는 뜻입니다.

```text
높은 음역을 더 선호한다.
빠른 tempo를 사용한다.
짧은 note를 선호하지만 1.0 길이 note도 허용한다.
강한 velocity를 선호한다.
상행 도약을 어느 정도 선호한다.
C-F-G-C chord progression에 맞는 melody note를 보상한다.
같은 음 반복이나 짧은 loop는 강하게 벌점 처리한다.
```

반대로 `sad` mode는 다음과 같은 profile을 사용합니다.

```text
scale: [57, 59, 60, 62, 64, 65, 67, 69]
tempo: 72
instrument: Acoustic Grand Piano
chord progression: Am - F - C - G
preferred durations: 1.0, 1.5
avoid durations: 0.25
preferred velocities: 48, 64
avoid velocities: 120
range: low_mid
contour: down
motion: stepwise
tension: low
```

즉 `sad`는 낮거나 중간 음역, 느린 note duration, 낮은 velocity, 하행 진행을 더 선호합니다.

### State

DQN 계열에서 state는 vector입니다.

```text
state = progress + bar_position + action_history + mood_vector
```

수식으로 쓰면 다음과 같습니다.

$$
s_t =
\left[
\frac{t}{T},
\frac{bar(t)}{3},
h_t,
\mathbf{q}_{mood}
\right]
$$

여기서:

```text
t: 현재 step
T: melody_length
bar(t): 4박 기준 현재 위치
h_t: padding된 action history
q_mood: 7-class mood vector
```

`history_size`가 `melody_length`와 같으면, 현재 episode 안에서 지금까지 선택한 action 전체가 state에 들어갑니다. 이 덕분에 reward가 과거 음표 패턴을 참고하더라도 MDP 구조로 설명할 수 있습니다.

각 항목의 의미는 다음과 같습니다.

`progress`는 현재 melody가 전체 길이 중 어디까지 진행되었는지를 나타냅니다.

$$
progress = \frac{t}{T}
$$

예를 들어 `melody_length = 32`이고 현재 8번째 음표를 만들 차례라면:

```text
progress = 8 / 32 = 0.25
```

이 값은 agent에게 곡의 초반, 중반, 후반 중 어디에 있는지 알려줍니다. 음악에서는 위치에 따라 좋은 선택이 달라질 수 있습니다.

```text
초반: 분위기를 여는 음
중반: 움직임과 변화
끝: 안정적인 마무리
```

`bar_position`은 현재 step이 4박 기준 마디 안에서 어디에 있는지를 나타냅니다.

$$
bar\_position = \frac{step\_count \bmod 4}{3}
$$

현재 구현은 4개의 event를 하나의 마디처럼 보고 있습니다.

```text
step_count = 0 -> bar_position = 0.000
step_count = 1 -> bar_position = 0.333
step_count = 2 -> bar_position = 0.667
step_count = 3 -> bar_position = 1.000
step_count = 4 -> bar_position = 0.000
```

이 정보가 필요한 이유는 마디 끝에서 안정음이나 chord tone으로 가는 선택이 음악적으로 더 자연스러울 수 있기 때문입니다. 실제 reward에도 `bar_end`, `harmony`, `final` 같은 항목이 있어, agent가 마디 위치를 알고 있어야 이런 구조를 학습할 수 있습니다.

`action_history`는 지금까지 agent가 고른 action들의 기록입니다. 예를 들어 현재까지 다음 action을 골랐다고 하겠습니다.

```text
actions = [103, 84, 107, 102]
```

DQN network는 항상 같은 길이의 input vector를 받아야 하므로, 아직 생성하지 않은 부분은 padding으로 채웁니다. 예를 들어 `melody_length = 8`, `history_size = 8`이면:

```text
raw history:
[103, 84, 107, 102]

padded history:
[103, 84, 107, 102, -1, -1, -1, -1]
```

이 값은 그대로 들어가지 않고 action size 기준으로 정규화됩니다.

$$
history\_value = \frac{action + 1}{action\_size}
$$

여기서 `+1`을 하는 이유는 padding 값이 `-1`이기 때문입니다.

```text
padding -1 -> (-1 + 1) / action_size = 0
real action 0 -> (0 + 1) / action_size
real action 103 -> (103 + 1) / action_size
```

이 history가 중요한 이유는 현재 reward가 과거 패턴을 많이 보기 때문입니다.

```text
같은 음을 너무 반복했는가?
ABAB loop가 생겼는가?
4음 motif가 과하게 반복되는가?
phrase가 너무 비슷한가?
```

이런 문제는 현재 action 하나만 보면 알 수 없습니다. 이전 action history가 state에 들어가야 agent가 방금까지 어떤 음악적 선택을 했는지 알고 다음 action을 고를 수 있습니다.

`mood_vector`는 classifier 결과를 음악 mood 공간으로 변환한 7-class 확률 벡터입니다.

예를 들어 이미지가 `happy`와 `excited` 사이의 분위기라면 다음과 같은 vector가 될 수 있습니다.

```text
mood_vector = [
  happy: 0.60,
  neutral: 0.05,
  sad: 0.01,
  dark: 0.02,
  angry: 0.02,
  excited: 0.28,
  unstable: 0.02
]
```

이렇게 전체 확률 분포를 state에 넣는 이유는, 이미지가 반드시 하나의 감정만 갖는 것은 아니기 때문입니다. 예를 들어 `happy = 0.60`, `excited = 0.28`이라면 단순히 `happy`라고만 보는 것보다 “밝지만 약간 신나는 분위기”라는 정보를 더 많이 전달할 수 있습니다.

전체 state 예시는 다음과 같습니다.

```text
melody_length = 8
action_size = 160
current step t = 4
actions = [103, 84, 107, 102]
mood_vector = [0.60, 0.05, 0.01, 0.02, 0.02, 0.28, 0.02]
```

그러면:

```text
progress = 4 / 8 = 0.5
bar_position = (4 % 4) / 3 = 0.0
```

history는 다음처럼 정규화됩니다.

```text
[103, 84, 107, 102, -1, -1, -1, -1]
-> [0.650, 0.531, 0.675, 0.644, 0, 0, 0, 0]
```

따라서 최종 state는 다음과 같은 숫자 벡터가 됩니다.

```text
[
  0.500,  # progress
  0.000,  # bar_position
  0.650,
  0.531,
  0.675,
  0.644,
  0,
  0,
  0,
  0,
  0.60,
  0.05,
  0.01,
  0.02,
  0.02,
  0.28,
  0.02
]
```

이 state가 의미하는 바는 다음과 같습니다.

```text
지금 곡의 50% 지점이고,
마디 시작 위치이며,
방금까지 이런 action sequence를 만들었고,
이미지 분위기는 happy가 가장 강하지만 excited도 꽤 있다.
```

즉 state는 단순한 숫자 배열이지만, 음악 생성 관점에서는 **현재까지 만들어진 음악의 문맥 + 감정 조건**을 DQN이 읽을 수 있게 압축한 표현입니다.

### Transition

한 step에서 agent가 action을 고르면 환경은 다음 작업을 합니다.

```text
1. action index를 pitch, duration, velocity로 decode
2. 해당 event를 melody에 append
3. reward breakdown 계산
4. step_count 증가
5. 다음 state 반환
```

강화학습 관점에서는 다음 전이입니다.

$$
(s_t, a_t) \rightarrow (s_{t+1}, r_t)
$$

뜻은 다음과 같습니다.

```text
현재 state s_t에서
agent가 action a_t를 선택하면
environment가 reward r_t를 계산하고
다음 state s_{t+1}을 반환한다.
```

우리 프로젝트에서는 이 transition이 **음표 event 하나를 생성하는 과정**입니다. 코드에서는 `MelodyEnv.step(action)`이 이 역할을 합니다.

한 step의 전체 흐름은 더 자세히 쓰면 다음과 같습니다.

```text
1. agent가 현재 state를 받는다.
2. agent가 action index 하나를 선택한다.
3. MelodyEnv가 action index를 pitch, duration, velocity로 decode한다.
4. decode된 값을 MusicEvent로 만든다.
5. MusicEvent를 현재 melody에 추가한다.
6. 방금 추가된 event를 기준으로 reward breakdown을 계산한다.
7. step_count를 1 증가시킨다.
8. melody_length에 도달했는지 확인해 done을 계산한다.
9. 업데이트된 melody history로 next_state를 만든다.
10. next_state, reward, done, info를 반환한다.
```

예를 들어 현재 state가 다음 의미를 가진다고 해보겠습니다.

```text
현재 곡의 50% 지점
마디 시작 위치
지금까지 action sequence = [103, 84, 107, 102]
mood = excited 중심
```

이 상태가 `s_t`입니다. agent는 이 state를 보고 다음 action을 하나 고릅니다.

```text
a_t = 109
```

이 action은 단순한 정수지만, environment는 이를 세 factor로 해석합니다.

```text
pitch_action = 5
duration_action = 1
velocity_action = 4
```

`excited` mode에서 이 값이 다음 event로 해석된다고 하겠습니다.

```text
pitch = 72      # C5
duration = 0.5
velocity = 120
```

그러면 environment는 다음과 같은 음악 event를 만듭니다.

```text
MusicEvent(
  tempo=168,
  position=현재 마디 위치,
  pitch=72,
  duration=0.5,
  velocity=120
)
```

이 event는 현재 melody 뒤에 붙습니다.

```text
이전 melody:
[E5, A4, E5, A4]

선택된 event:
C5, duration 0.5, velocity 120

transition 후 melody:
[E5, A4, E5, A4, C5]
```

그 다음 environment는 방금 선택한 event가 현재 문맥에서 얼마나 좋은지 reward를 계산합니다. `excited`라면 예를 들어 다음 항목들이 평가됩니다.

```text
interval:
이전 음 A4에서 C5로 이동이 자연스러운가?

repetition:
같은 음을 너무 반복하지 않았는가?

duration:
0.5 길이가 excited에 어울리는가?

velocity:
120 세기가 excited에 어울리는가?

harmony:
현재 chord와 C5가 잘 맞는가?

bar_end:
마디 끝이라면 안정적인 음인가?

mode:
excited답게 상행/도약 느낌이 있는가?
```

각 항목은 mood별 weight가 곱해진 뒤 합산됩니다.

$$
r_t = \sum_{c \in C} w_{mode,c} \cdot r_{t,c}
$$

이후 `step_count`가 증가합니다.

```text
step_count = step_count + 1
```

그리고 episode가 끝났는지 확인합니다.

```text
done = step_count >= melody_length
```

예를 들어 `melody_length = 32`이면 32개의 event를 만든 순간 episode가 끝납니다.

마지막으로 다음 state `s_{t+1}`을 만듭니다. 이전 action history가 다음과 같았다면:

```text
[103, 84, 107, 102]
```

새 action이 추가된 뒤에는 다음 history가 됩니다.

```text
[103, 84, 107, 102, 109]
```

progress도 업데이트됩니다.

```text
이전 progress = 4 / 32 = 0.125
다음 progress = 5 / 32 = 0.156
```

bar position도 업데이트됩니다.

```text
이전 step_count = 4 -> bar_position = 0.000
다음 step_count = 5 -> bar_position = 0.333
```

즉 `s_{t+1}`은 새 action history, 새 progress, 새 bar position, 같은 mood vector를 포함한 vector입니다.

DQN 학습 관점에서는 이 transition 하나가 replay buffer에 저장됩니다.

```text
(
  state,
  action,
  reward,
  next_state,
  done
)
```

예를 들어 다음과 같은 경험 하나가 저장될 수 있습니다.

```text
(
  s_t,
  109,
  2.85,
  s_{t+1},
  false
)
```

이 경험들이 buffer에 쌓이고, DQN은 나중에 random mini-batch로 뽑아 학습합니다.

현재 transition은 대부분 deterministic합니다. 같은 state에서 같은 action을 고르면 같은 pitch, duration, velocity가 추가되고, 같은 reward와 next state가 만들어집니다. 환경 자체의 randomness는 거의 없고, randomness는 주로 다음 부분에서 발생합니다.

```text
epsilon-greedy exploration
replay buffer sampling
초기 network weight
```

정리하면, 우리 프로젝트에서 transition은 다음 과정입니다.

```text
현재까지 만든 음악 상태 s_t
+ agent가 고른 다음 event action a_t
->
새로운 event가 melody에 추가됨
+ reward 계산됨
+ 다음 음악 상태 s_{t+1} 생성됨
```

음악적으로 말하면, transition은 **지금까지 만든 멜로디에 다음 음표 하나를 붙이고, 그 선택이 음악적으로 얼마나 괜찮았는지 평가한 뒤, 업데이트된 멜로디 문맥을 다음 state로 넘기는 과정**입니다.

### Reward

현재 reward는 여러 component의 합입니다.

$$
r_t = \sum_{c \in C} w_{mode,c} \cdot r_{t,c}
$$

여기서:

```text
C: reward component 집합
r_{t,c}: component c의 raw reward
w_{mode,c}: 현재 mood에서 component c에 주는 가중치
```

reward component는 다음과 같습니다.

```text
interval
repetition
short_loop
octave_jump
stable_note
bar_end
pattern
diversity
duration
duration_balance
velocity
harmony
range_profile
mode
final
```

예를 들어 `harmony` reward는 현재 chord template과 melody note의 pitch class가 맞는지 봅니다.

```text
현재 chord tone이면 보상
마디 끝에서 chord tone이면 더 큰 보상
scale 안의 non-chord tone이면 약한 감점
scale 밖이면 큰 감점
```

`repetition`, `short_loop`, `diversity`는 같은 음이나 같은 짧은 패턴을 반복하는 문제를 줄이기 위해 사용됩니다.

## 6. Factorized DQN Agent의 작동 과정

### DQN이란 무엇인가

DQN은 **Deep Q-Network**의 약자입니다. 강화학습에서 agent는 현재 state `s_t`를 보고 action `a_t`를 선택합니다. 그 action이 얼마나 좋은지를 나타내는 값을 Q-value라고 합니다.

$$
Q(s_t, a_t)
$$

Q-value는 다음 의미를 가집니다.

```text
현재 state에서 이 action을 선택했을 때,
앞으로 받을 reward의 누적 기대값
```

기본 Q-learning의 목표는 다음 Bellman equation을 만족하도록 Q-value를 학습하는 것입니다.

$$
Q(s_t, a_t)
\leftarrow
r_t + \gamma \max_{a'} Q(s_{t+1}, a')
$$

여기서 \(\gamma\)는 discount factor입니다.

DQN은 이 Q-value table을 neural network로 근사합니다.

$$
Q_{\theta}(s, a) \approx Q(s, a)
$$

즉 state를 network에 넣으면 가능한 action들의 Q-value가 나옵니다.

### Replay Buffer

DQN은 매 step에서 다음 경험을 저장합니다.

```text
state
action
reward
next_state
done
next_valid_actions
```

수식으로는 다음 tuple입니다.

$$
(s_t, a_t, r_t, s_{t+1}, done)
$$

이 경험들은 replay buffer에 저장되고, 학습할 때 random mini-batch로 샘플링됩니다. 이렇게 하는 이유는 연속된 sequence를 그대로 학습하면 데이터 사이 상관관계가 너무 강해져 학습이 불안정해지기 때문입니다.

### Target Network

DQN은 target network를 따로 둡니다.

```text
online Q network: 현재 학습되는 network
target Q network: 일정 주기마다 online network를 복사한 안정적인 network
```

target은 다음과 같이 계산됩니다.

$$
y_t =
r_t + \gamma \max_{a'} Q_{target}(s_{t+1}, a')
$$

여기서 `Q_target`은 target network가 계산한 Q-value입니다.

loss는 보통 다음 MSE입니다.

$$
loss =
\left(
Q_{online}(s_t, a_t) - y_t
\right)^2
$$

### Double DQN

현재 DQN 구현은 `use_double_dqn=True`가 기본입니다. Double DQN은 action 선택과 action 평가를 분리합니다.

일반 DQN은 target을 이렇게 잡습니다.

$$
y_t =
r_t + \gamma \max_{a'} Q_{target}(s_{t+1}, a')
$$

Double DQN은 다음처럼 합니다.

$$
a^*
=
\arg\max_{a'} Q_{online}(s_{t+1}, a')
$$

$$
y_t =
r_t + \gamma Q_{target}(s_{t+1}, a^*)
$$

즉 action 선택은 online network가 하고, 그 action의 Q-value 평가는 target network가 합니다. 이렇게 하면 Q-value가 과대평가되는 문제를 줄일 수 있습니다.

### 일반 DQN과 우리의 Factorized DQN의 차이

일반 DQN은 모든 action 조합을 하나의 action space로 봅니다.

현재 음악 action은 다음 세 요소로 구성됩니다.

```text
pitch
duration
velocity
```

만약 가능한 pitch가 8개, duration이 4개, velocity가 5개라면 전체 action 수는 다음과 같습니다.

$$
|\mathcal{A}| = 8 \times 4 \times 5 = 160
$$

일반 DQN은 state를 넣으면 160개 action 각각의 Q-value를 직접 출력합니다.

$$
Q(s, a_0), Q(s, a_1), \dots, Q(s, a_{159})
$$

하지만 음악적으로 보면 action은 하나의 덩어리가 아니라 event factor의 조합입니다.

```text
음높이 선택
길이 선택
세기 선택
```

그래서 현재 기본 agent는 Factorized DQN입니다.

```text
Pitch Q Network
Duration Q Network
Velocity Q Network
```

각 network는 같은 state를 입력받지만 서로 다른 factor의 Q-value를 출력합니다.

$$
Q_p(s, p)
$$

$$
Q_d(s, d)
$$

$$
Q_v(s, v)
$$

하나의 action \(a\)가 다음처럼 decode된다고 하겠습니다.

$$
a = (p, d, v)
$$

그러면 최종 action score는 세 Q-value의 평균입니다.

$$
Q_{factorized}(s, a)
=
\frac{
Q_p(s, p) + Q_d(s, d) + Q_v(s, v)
}{3}
$$

agent는 valid action들 중 이 값이 가장 큰 action을 선택합니다.

$$
a_t =
\arg\max_{a \in \mathcal{A}_{valid}}
Q_{factorized}(s_t, a)
$$

### 왜 Factorized DQN을 쓰는가

Factorized DQN을 쓰는 이유는 세 가지입니다.

첫째, action space가 커져도 구조적으로 확장하기 쉽습니다.

```text
현재: pitch x duration x velocity
다음: pitch x duration x velocity x chord
이후: pitch x duration x velocity x chord x instrument
```

둘째, 음악 event의 의미와 잘 맞습니다. 음악을 만들 때 pitch, duration, velocity는 서로 관련은 있지만 완전히 같은 종류의 결정은 아닙니다.

셋째, 우리가 구상한 event-token 기반 생성 구조로 확장하기 쉽습니다. 다음 단계에서는 `Tempo`, `Bar`, `Position`, `Pitch`, `Duration`, `Velocity` 같은 event representation을 더 명시적으로 다룰 수 있습니다. 현재 factorized action 구조는 그 방향으로 넘어가기 위한 중간 단계입니다.

## 7. Action에 대한 자세한 설명

현재 action은 하나의 정수 index로 저장됩니다. 하지만 그 정수는 실제로 다음 세 요소를 encoding한 값입니다.

```text
pitch_action
duration_action
velocity_action
```

기본 후보는 다음과 같습니다.

```text
durations = [0.25, 0.5, 1.0, 1.5]
velocities = [48, 64, 88, 104, 120]
```

pitch 후보는 mood별 scale에서 결정됩니다.

예를 들어 `happy`는 C major 계열입니다.

```text
happy notes = [60, 62, 64, 65, 67, 69, 71, 72]
```

MIDI note로 보면 다음과 같습니다.

```text
60: C4
62: D4
64: E4
65: F4
67: G4
69: A4
71: B4
72: C5
```

`excited`는 더 높은 음역을 포함합니다.

```text
excited notes = [60, 62, 64, 67, 69, 72, 74, 76]
```

### Action index를 세 개의 factor로 decode하는 방식

Environment는 agent가 고른 action을 처음에는 하나의 정수로 받습니다.

예를 들어 agent가 다음 action을 선택했다고 하겠습니다.

```text
a_t = 109
```

하지만 이 값 109 자체가 바로 음 하나를 의미하는 것은 아닙니다. 이 정수 안에는 다음 세 가지 선택이 함께 들어 있습니다.

```text
pitch_action: 몇 번째 pitch 후보를 고를 것인가
duration_action: 몇 번째 duration 후보를 고를 것인가
velocity_action: 몇 번째 velocity 후보를 고를 것인가
```

즉 우리 action은 원래 3차원 선택입니다.

```text
(pitch_action, duration_action, velocity_action)
```

다만 DQN의 action space는 보통 하나의 discrete index로 다루는 것이 편하기 때문에, 이 3차원 선택을 하나의 정수 action index로 펼쳐서 저장합니다. Environment는 이 정수를 다시 세 개의 factor로 풀어냅니다.

현재 기본 설정은 다음과 같습니다.

```text
duration 후보 개수 N_d = 4
velocity 후보 개수 N_v = 5
```

따라서 하나의 pitch마다 가능한 rhythm 조합 수는 다음과 같습니다.

$$
N_{rhythm} = N_d \times N_v = 4 \times 5 = 20
$$

즉 action index는 20개 단위로 pitch 구간이 나뉩니다.

```text
action 0  ~ 19   -> pitch_action 0
action 20 ~ 39   -> pitch_action 1
action 40 ~ 59   -> pitch_action 2
action 60 ~ 79   -> pitch_action 3
action 80 ~ 99   -> pitch_action 4
action 100 ~ 119 -> pitch_action 5
...
```

그래서 `a_t = 109`는 `100 ~ 119` 구간에 있으므로 `pitch_action = 5`가 됩니다.

코드에서는 다음 방식으로 계산합니다.

$$
pitch\_action =
\left\lfloor
\frac{a}{N_d N_v}
\right\rfloor
$$

$$
rhythm\_action =
a \bmod (N_d N_v)
$$

$$
duration\_action =
\left\lfloor
\frac{rhythm\_action}{N_v}
\right\rfloor
$$

$$
velocity\_action =
rhythm\_action \bmod N_v
$$

여기서:

```text
N_d: duration 후보 개수
N_v: velocity 후보 개수
```

`a_t = 109`를 실제로 계산하면 다음과 같습니다.

먼저 pitch를 고릅니다.

$$
pitch\_action =
\left\lfloor
\frac{109}{4 \times 5}
\right\rfloor
=
\left\lfloor
\frac{109}{20}
\right\rfloor
= 5
$$

그다음 pitch를 고르고 남은 rhythm 부분을 구합니다.

$$
rhythm\_action =
109 \bmod 20 = 9
$$

이 `rhythm_action = 9` 안에는 duration 선택과 velocity 선택이 같이 들어 있습니다. velocity 후보가 5개이므로, 5개 단위로 duration 구간이 나뉩니다.

```text
rhythm_action 0 ~ 4   -> duration_action 0
rhythm_action 5 ~ 9   -> duration_action 1
rhythm_action 10 ~ 14 -> duration_action 2
rhythm_action 15 ~ 19 -> duration_action 3
```

따라서 `rhythm_action = 9`는 `duration_action = 1`에 해당합니다.

수식으로는 다음과 같습니다.

$$
duration\_action =
\left\lfloor
\frac{9}{5}
\right\rfloor
= 1
$$

그리고 velocity는 나머지로 결정됩니다.

$$
velocity\_action =
9 \bmod 5 = 4
$$

따라서 `a_t = 109`는 다음 factor로 decode됩니다.

```text
pitch_action = 5
duration_action = 1
velocity_action = 4
```

`excited` mood에서는 pitch 후보가 다음과 같습니다.

```text
excited notes = [60, 62, 64, 67, 69, 72, 74, 76]
durations = [0.25, 0.5, 1.0, 1.5]
velocities = [48, 64, 88, 104, 120]
```

그러면 각 factor는 실제 음악 event로 다음처럼 바뀝니다.

```text
pitch = notes[5] = 72
duration = durations[1] = 0.5
velocity = velocities[4] = 120
```

MIDI note 72는 C5입니다. 즉 `a_t = 109`는 `excited` mood에서 다음 event가 됩니다.

```text
C5, duration 0.5, velocity 120
```

반대로 세 factor를 다시 하나의 action index로 encode할 수도 있습니다.

$$
rhythm\_action =
duration\_action \cdot N_v + velocity\_action
$$

$$
a =
pitch\_action \cdot (N_dN_v) + rhythm\_action
$$

위 예시를 다시 encode하면 다음과 같습니다.

$$
rhythm\_action = 1 \cdot 5 + 4 = 9
$$

$$
a = 5 \cdot (4 \cdot 5) + 9 = 109
$$

## 8. Reward 설계

현재 reward는 rule-based입니다. 하지만 단순히 하나의 기준만 보지 않고, 여러 음악적 기준을 component로 나누어 계산합니다.

### Interval

직전 음과 현재 음 사이의 거리입니다.

```text
같은 음: 감점
작은 이동: 보상
너무 큰 이동: 감점
```

### Repetition

같은 pitch가 연속으로 반복될수록 강하게 감점합니다.

```text
2번 반복: -2.5
3번 반복: -7.0
4번 이상 반복: -13.0
```

### Short Loop

`ABAB`, `ABABAB` 같은 짧은 loop를 감점합니다.

### Pattern

4음 motif는 약하게 허용하지만, 같은 motif가 계속 반복되면 감점합니다. 현재는 반복 motif를 큰 보상으로 보지 않습니다. 이유는 이전 실험에서 pattern reward가 과하면 같은 구절을 계속 반복하는 문제가 생겼기 때문입니다.

### Diversity

episode 마지막에 사용한 pitch 다양성, 같은 음 연속 비율, 2-gram/4-gram 반복 횟수를 보고 감점합니다.

### Duration / Duration Balance

`duration`은 현재 note 하나가 mood에 어울리는 길이인지 봅니다.

`duration_balance`는 전체 melody가 너무 한 종류의 rhythm에 치우치지 않았는지 봅니다. 예를 들어 `excited`라도 모든 음이 0.25 또는 0.5만 있으면 기계적으로 들릴 수 있으므로 감점합니다.

### Velocity

velocity가 mood에 어울리는지 평가합니다.

예를 들어:

```text
excited: 104, 120 선호
sad: 48, 64 선호
```

### Harmony

현재 chord progression에 맞는 note인지 평가합니다.

```text
chord tone: 보상
bar end chord tone: 더 큰 보상
scale tone but non-chord tone: 약한 감점
non-scale tone: 큰 감점
```

### Mode

mood profile의 contour, motion, tension을 반영합니다.

예를 들어 `excited`는 상행 도약을 어느 정도 선호합니다.

```text
current_note > previous_note and interval >= 4
```

이면 추가 보상을 받을 수 있습니다.

### Mood별 Reward Weight

같은 component라도 mood에 따라 중요도가 다릅니다.

예를 들어 `excited`는 반복이 매우 치명적인 실패 양상이므로 다음 weight가 큽니다.

```text
repetition
short_loop
pattern
diversity
duration_balance
```

반대로 `sad`와 `dark`는 느린 duration, 낮은 음역, 하행 진행이 더 중요합니다.

이 구조는 다음 수식으로 요약할 수 있습니다.

$$
r_t = \sum_{c \in C} w_{mode,c}r_{t,c}
$$

## 9. MIDI Export

학습된 agent가 melody event sequence를 만들면 MIDI로 저장합니다.

현재 MIDI는 세 track으로 구성됩니다.

```text
melody track: agent가 생성한 pitch, duration, velocity
chord pad track: mood별 chord progression을 길게 깔아주는 pad
bass track: chord root를 낮은 음역에서 받쳐주는 bass
```

예를 들어 `happy`는 다음 chord progression을 사용합니다.

```text
C - G - Am - F
```

`excited`는 다음 progression을 사용합니다.

```text
C - F - G - C
```

MIDI 저장 시 mood profile의 tempo와 instrument도 반영됩니다.

```text
happy: tempo 132, Bright Acoustic Piano
sad: tempo 72, Acoustic Grand Piano
excited: tempo 168, Electric Piano 1
angry: tempo 150, Rock Organ
unstable: tempo 124, Clavinet
```

## 10. User Feedback Loop

현재 프로젝트에는 로컬 feedback GUI가 포함되어 있습니다.

실행:

```bash
python3 RL/feedback_gui.py
```

접속:

```text
http://127.0.0.1:7860
```

GUI에서는 다음 작업을 할 수 있습니다.

```text
1. 이미지 업로드
2. 음악 생성
3. Web Audio preview 재생
4. MIDI 파일 다운로드
5. 5점 척도 피드백 저장
```

피드백 항목은 다음과 같습니다.

```text
emotion_match
naturalness
repetition_control
richness
overall
```

피드백은 다음 위치에 저장됩니다.

```text
results/feedback/feedback_log.jsonl
results/feedback/feedback_log.csv
```

또한 피드백은 다음 generation에 사용할 reward weight override를 업데이트합니다.

```text
results/feedback/reward_weight_overrides.json
```

예를 들어 사용자가 `repetition_control`을 낮게 평가하면, 해당 mood의 다음 weight가 조금 올라갑니다.

```text
repetition
short_loop
diversity
pattern
```

즉 현재 user feedback은 아직 neural preference model은 아니지만, rule reward를 조금씩 조정하는 첫 단계입니다.

## 11. Selection Score

단순 reward만으로 best checkpoint를 고르면 문제가 생길 수 있습니다.

예를 들어 chord tone을 많이 쓰고 bar end가 안정적이면 reward는 높지만, 실제로는 같은 음만 반복하는 melody일 수 있습니다.

그래서 현재 DQN/factorized DQN은 training 중 best checkpoint를 고를 때 `selection_score`를 사용합니다.

$$
\begin{aligned}
selection
=&\ reward
+ 0.45 \cdot melodic\_quality \\
&- 35 \cdot same\_adjacent\_ratio \\
&- 18 \cdot phrase\_repeat\_ratio \\
&- 28 \cdot short\_note\_excess \\
&- 2 \cdot \max(0, max\_4gram\_count - 2)
\end{aligned}
$$

이 score는 다음 요소를 함께 봅니다.

```text
생성 reward
melodic quality
같은 음 연속 비율
phrase 반복 비율
짧은 음 과다 사용
4-gram 반복
```

GUI feedback이 들어오면 feedback-adjusted selection score도 계산됩니다.

$$
selection_{feedback}
= selection + feedback\_score
$$

## 12. 현재 구조와 다음 단계 Architecture의 차이

우리가 다음 단계로 구상하는 architecture는 다음 요소를 포함합니다.

```text
Q Network: Linear Transformer
Target Q Network: Linear Transformer
Agent trajectory
Expert trajectory
Replay memory
AIRL Network: Longformer
Learned reward
```

우리 프로젝트의 현재 상태는 다음과 같습니다.

```text
있는 것:
- MelodyEnv
- DQN / Factorized DQN
- Target network
- Replay buffer
- Agent trajectory / generation log
- Rule-based reward
- User feedback log
- Feedback-based reward weight tuning

아직 없는 것:
- Expert MIDI trajectory dataset
- Transformer Q network
- AIRL / Longformer reward network
- Learned reward model
- A/B preference training
```

즉 현재 구조는 DQN loop와 환경 부분을 먼저 갖춘 상태이고, learned reward와 Transformer 기반 sequence modeling은 user feedback log와 generation trajectory를 쌓으면서 준비하는 단계입니다.

## 13. 실행 예시

기본 이미지와 기본 classifier로 단일 생성:

```bash
python3 RL/main.py
```

특정 이미지를 넣어서 생성:

```bash
python3 RL/main.py --image path/to/image.jpg
```

짧게 동작 확인:

```bash
python3 RL/main.py --episodes 3 --melody-length 4 --no-midi
```

excited만 batch 생성:

```bash
python3 RL/run_mood_batch.py --moods excited --samples 2 --episodes 5000 --melody-length 32
```

여러 mood 비교:

```bash
python3 RL/run_mood_batch.py --moods happy excited sad dark unstable --samples 2 --episodes 5000 --melody-length 32
```

feedback GUI 실행:

```bash
python3 RL/feedback_gui.py
```

## 14. 결과 파일

일반 실험 결과는 다음 위치에 저장됩니다.

```text
results/experiments/
```

각 실험 폴더에는 보통 다음 파일이 생깁니다.

```text
summary.json
episode_rewards.csv
episode_losses.csv
generated_reward_breakdown.csv
generated_*.mid
```

GUI generation 결과는 다음 위치에 저장됩니다.

```text
results/feedback_gui/generations/
```

피드백 데이터는 다음 위치에 저장됩니다.

```text
results/feedback/feedback_log.jsonl
results/feedback/feedback_log.csv
results/feedback/reward_weight_overrides.json
```

## 15. 앞으로의 발전 방향

현재 다음 발전 방향을 목표로 합니다.

```text
1. feedback 중복 저장 방지
2. 같은 이미지에서 여러 sample 생성 후 A/B preference 수집
3. generation trajectory를 AIRL/Transformer 학습용 형태로 정리
4. user feedback 기반 preference reward model 학습
5. MIDI dataset 또는 high-rated generated MIDI를 expert trajectory로 사용
6. Factorized DQN backbone을 Transformer 기반 Q network로 확장
7. 최종적으로 AIRL-like learned reward를 도입
```

장기적으로는 현재 rule reward 중심 구조를 다음 구조로 확장하는 것이 목표입니다.

```text
rule reward
-> feedback-adjusted selection
-> preference reward model
-> expert/preference trajectory replay
-> Transformer Q network
-> AIRL-like learned reward
```
