# Emotion Classification

이미지에서 감정을 분류하고, 결과를 **음악 생성용 7가지 무드**로 변환하는 프로젝트입니다.

[Hugging Face EmoSet-118K](https://huggingface.co/datasets/Woleek/EmoSet-118K) 데이터셋을 기반으로 **EfficientNet-B0**를 파인튜닝합니다.

---

## 개요

| 단계 | 설명 |
|------|------|
| 감정 분류 | 8개 EmoSet 감정 클래스 예측 |
| 무드 매핑 | 예측 확률을 7개 음악 무드로 변환 |

### EmoSet 감정 (8클래스)

`amusement` · `anger` · `awe` · `contentment` · `disgust` · `excitement` · `fear` · `sadness`

### 음악 무드 (7클래스)

`HAPPY` · `NEUTRAL` · `SAD` · `DARK` · `ANGRY` · `EXCITED` · `UNSTABLE`

`predict.py`에서 EmoSet 확률을 가중 합산해 무드 벡터로 변환합니다.

---

## 프로젝트 구조

```
emotion_classification/
├── make_emoset_24k.py   # HF 데이터셋 → 로컬 이미지 폴더 생성
├── data_loaders.py      # DataLoader 및 전처리
├── model.py             # EfficientNet-B0 모델 정의
├── train.py             # 학습 및 평가
├── predict.py           # 단일 이미지 추론 + 무드 변환
├── check_counts.py      # split별 이미지 개수 확인
├── requirements.txt
├── datasets/            # (로컬 생성, Git 미포함)
│   └── EmoSet_24K_split/
│       ├── train/
│       ├── val/
│       └── test/
└── checkpoints/         # (학습 후 생성, Git 미포함)
    ├── best.pt
    └── history.jsonl
```

---

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**요구 사항:** Python 3.10+, PyTorch 2.0+, CUDA(선택)

---

## 데이터셋 준비

EmoSet-118K에서 클래스당 **3,000장**을 샘플링해 총 **24,000장**을 저장합니다.

| split | 비율 | 클래스당 장수 |
|-------|------|----------------|
| train | 70% | 2,100 |
| val   | 15% | 450 |
| test  | 15% | 450 |

```bash
python make_emoset_24k.py
```

생성 경로: `datasets/EmoSet_24K_split/{train,val,test}/{클래스명}/*.jpg`

개수 확인:

```bash
python check_counts.py
```

---

## 학습

```bash
python train.py
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 20 | 학습 에폭 수 |
| `--batch-size` | 32 | 배치 크기 |
| `--lr` | 5e-5 | 학습률 |
| `--patience` | 5 | Early stopping (0이면 비활성) |
| `--seed` | 42 | 랜덤 시드 |

예시:

```bash
python train.py --epochs 30 --batch-size 16 --patience 7
```

### 출력

- `checkpoints/best.pt` — 검증 정확도가 가장 높은 체크포인트
- `checkpoints/history.jsonl` — 에폭별 train/val loss·accuracy 기록

학습 종료 후 **test set**으로 최종 평가를 수행합니다.

---

## 추론

```bash
python predict.py --image test.jpg
```

체크포인트 경로 지정:

```bash
python predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pt
```

### 출력 예시

1. **EmoSet 감정 확률** (8클래스, 내림차순)
2. **음악 무드 확률** (7클래스, 내림차순)
3. **최종 무드** (`Final mood: HAPPY` 등)

---

## 모델

- 백본: **EfficientNet-B0** (ImageNet 사전학습 가중치 사용)
- 입력: 224×224 RGB, ImageNet 정규화
- 학습 시 증강: RandomHorizontalFlip, RandomRotation, ColorJitter

---

## 참고

- `datasets/`, `checkpoints/`는 용량이 커서 저장소에 포함하지 않는 것을 권장합니다.
- 데이터셋은 `make_emoset_24k.py` 실행으로 로컬에서 재생성할 수 있습니다.
- 원본 데이터셋: [Woleek/EmoSet-118K](https://huggingface.co/datasets/Woleek/EmoSet-118K)
