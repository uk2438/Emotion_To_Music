# EmotionRL Composer

얼굴 표정 이미지를 입력으로 받아 감정을 분석하고, 강화학습(Q-learning)을 이용해 감정에 맞는 음악을 생성하는 프로젝트입니다.

## 전체 시스템 흐름

이미지 입력 → DeepFace 감정 분석 → 감정 label을 음악 mode로 변환 → Q-learning 기반 멜로디 생성 → MIDI 파일 출력

## 프로젝트 목표

본 프로젝트의 목표는 사용자의 얼굴 표정에서 감정을 추론하고, 그 감정을 음악적 특징으로 변환하여 강화학습 에이전트가 감정에 맞는 멜로디를 생성하도록 하는 것입니다.

예를 들어 웃는 얼굴이 입력되면 happy로 분류하고, 밝고 상승감 있는 멜로디를 생성합니다. 혐오 표정이 입력되면 disgust로 분류하고, 이를 unstable 음악 mode로 변환하여 불안정한 느낌의 멜로디를 생성합니다.

## 현재 프로젝트 구조

```text
RL project/
├── emotion_rl_env/          # Python 가상환경
├── face/                    # 테스트 얼굴 이미지 폴더
│   └── disgust.jpg
├── RL/
│   ├── main.py              # 전체 실행 파일
│   ├── melody_env.py        # 강화학습 환경
│   ├── q_learning_agent.py  # Q-learning 에이전트
│   └── readme.md            # 이 파일
└── README.md
```

## 코드 상세 설명

### main.py

`main.py`는 프로젝트의 메인 실행 파일로, 전체 시스템의 흐름을 제어합니다.

#### 주요 기능
- **얼굴 감정 분석**: `analyze_face_emotion()` 함수를 통해 DeepFace 라이브러리를 사용하여 얼굴 이미지를 분석하고, 감정을 추출합니다. 한글 경로 지원을 위해 `load_image_safely()` 함수로 이미지를 안전하게 로드합니다.
- **감정 매핑**: `map_emotion_to_music_mode()` 함수로 DeepFace의 감정 레이블을 음악 모드로 변환합니다. 예: "happy" → "happy", "disgust" → "unstable".
- **에이전트 학습**: `train_agent()` 함수로 지정된 감정 모드에 대해 Q-learning 에이전트를 학습시킵니다. 에피소드 수와 멜로디 길이를 설정할 수 있습니다.
- **멜로디 생성**: `generate_melody()` 함수로 학습된 에이전트를 사용하여 탐험 없이 그리디 방식으로 멜로디를 생성합니다.
- **MIDI 출력**: `export_melody_to_midi()` 함수로 생성된 멜로디를 MIDI 파일로 저장합니다. pretty_midi 라이브러리를 사용합니다.
- **노트 변환**: `convert_to_note_names()` 함수로 MIDI 피치를 읽기 쉬운 음 이름으로 변환합니다.

#### 실행 흐름
1. 얼굴 이미지 경로를 설정합니다 (기본: `face/disgust.jpg`).
2. 감정을 분석하고 음악 모드를 결정합니다.
3. Q-learning 에이전트를 학습시킵니다 (기본: 5000 에피소드, 멜로디 길이 32).
4. 학습된 에이전트로 멜로디를 생성합니다.
5. 생성된 멜로디를 콘솔에 출력하고 MIDI 파일로 저장합니다.

### melody_env.py

`melody_env.py`는 강화학습 환경을 구현한 파일로, 멜로디 생성 과정을 환경으로 모델링합니다.

#### 클래스: MelodyEnv
- **초기화**: 감정 모드와 멜로디 길이를 받아 환경을 설정합니다. 모드에 따라 사용할 음계를 선택합니다.
- **음계 선택**: `_select_scale()` 메서드로 모드별 음계를 정의합니다.
  - happy/neutral: C major (도, 레, 미, 파, 솔, 라, 시, 높은 도)
  - sad/dark: A minor (라, 시, 도, 레, 미, 파, 솔, 높은 라)
  - angry: D minor 느낌 (레, 미, 파, 솔, 라, 라#, 높은 도, 높은 레)
  - excited: 밝고 높은 음역 (도, 레, 미, 솔, 라, 높은 도, 높은 레, 높은 미)
  - unstable: 반음 포함 불안정한 음계 (도, 도#, 미, 파#, 솔, 라#, 시, 높은 도)

#### 주요 메서드
- **reset()**: 새로운 에피소드를 시작합니다. 액션과 멜로디 리스트를 초기화합니다.
- **step(action)**: 에이전트의 액션을 받아 다음 상태, 보상, 종료 여부, 정보를 반환합니다.
- **_get_state()**: 현재 상태를 반환합니다. 상태는 (직전 액션 인덱스, 현재 위치) 튜플입니다.
- **_calculate_reward(action)**: 여러 음악적 기준으로 보상을 계산합니다.
  - 간격 보상: 직전 음과의 거리로 자연스러움 평가
  - 반복 페널티: 같은 음 3번 연속 반복 방지
  - 짧은 루프 페널티: ABAB 패턴 과도 반복 방지
  - 안정음 보상: 1도, 3도, 5도, 옥타브 음 선호
  - 마디 끝 보상: 4음마다 안정적으로 끝나면 보상
  - 패턴 보상: 반복 모티프 형성 시 보상
  - 모드 보상: 감정 모드별 특징 반영 (예: happy는 상행 선호)
  - 최종 보상: 멜로디 끝이 모드에 맞게 끝나면 추가 보상

### q_learning_agent.py

`q_learning_agent.py`는 Q-learning 알고리즘을 구현한 에이전트 클래스입니다.

#### 클래스: QLearningAgent
- **초기화**: 액션 크기와 하이퍼파라미터를 설정합니다.
  - alpha (학습률): 0.1
  - gamma (할인율): 0.95
  - epsilon (탐험률): 초기 1.0, 점차 감소
  - epsilon_decay: 0.995
  - epsilon_min: 0.05

#### 주요 메서드
- **choose_action(state, training)**: epsilon-greedy 방식으로 액션을 선택합니다. 학습 시 탐험, 추론 시 최적 액션 선택.
- **update(state, action, reward, next_state, done)**: Q-learning 업데이트를 수행합니다. Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **decay_epsilon()**: 에피소드마다 epsilon을 감소시켜 탐험을 줄입니다.

#### Q-table 구조
- defaultdict를 사용하여 상태별 액션 값 배열을 저장합니다.
- 상태: (직전 액션 인덱스, 현재 위치)
- 액션: 음계 내 음의 인덱스

## 사용 방법

1. 가상환경 활성화: `source emotion_rl_env/bin/activate`
2. 실행: `python RL/main.py`
3. 결과: 생성된 MIDI 파일이 프로젝트 루트에 저장됩니다.

## 의존성

- deepface: 얼굴 감정 분석
- opencv-python: 이미지 처리
- numpy: 수치 계산
- pretty_midi: MIDI 파일 생성
- tensorflow/keras: DeepFace 내부 사용

## 향후 개선 방향

- 더 다양한 감정 모드 추가
- 보상 함수 세부 튜닝
- 긴 멜로디 생성 지원
- 실시간 얼굴 인식 연동

