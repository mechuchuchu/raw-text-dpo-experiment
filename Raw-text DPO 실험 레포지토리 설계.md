# Raw-text DPO 실험 레포지토리 설계

## 1. 목적

`allenai/Dolci-Instruct-DPO`의 preference pair를 일반적인 raw text sequence로 직렬화해 LoRA 학습한 뒤, chosen/rejected likelihood 차이가 나타나는지 확인한다.

실험은 다음 효과를 분리해서 측정할 수 있어야 한다.

1. 답변 내용에 따른 likelihood 차이
2. `###chosen` 및 `###rejected` 라벨에 따른 위치·라벨 효과
3. sequence 안에서 chosen/rejected가 먼저 또는 나중에 등장하는 위치 효과
4. Base model과 LoRA model 사이의 변화

## 2. 권장 레포지토리 구조

```text
raw-text-dpo-experiment/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   ├── sanity.yaml
│   └── main.yaml
├── src/
│   └── rawtext_dpo/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── serialization.py
│       ├── tokenization.py
│       ├── training.py
│       ├── evaluation.py
│       ├── metrics.py
│       └── artifacts.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_subset.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── summarize_results.py
├── tests/
│   ├── test_data.py
│   ├── test_serialization.py
│   └── test_metrics.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── outputs/
│   ├── checkpoints/.gitkeep
│   ├── results/.gitkeep
│   └── logs/.gitkeep
└── reports/
    └── figures/.gitkeep
```

대용량 데이터, 모델 checkpoint, tokenized cache, 결과 CSV는 Git에 넣지 않는다. `data/`, `outputs/`, Hugging Face cache는 `.gitignore`로 제외하고, 각 run의 설정과 요약 결과만 보존한다.

## 3. 모듈 책임

### `data.py`

- Hugging Face 데이터셋 로드
- `chosen`/`rejected` 메시지 리스트에서 단일 턴 또는 멀티턴 데이터 정규화
- `None` 또는 빈 assistant 답변 제거
- `prompt_id` 기반 train/evaluation 분할
- seed와 데이터셋 revision 기록
- 샘플 manifest 생성

초기 sanity check에서는 다음 조건만 허용한다.

```text
chosen  = [user, assistant]
rejected = [user, assistant]
모든 content가 비어 있지 않은 문자열
```

### `serialization.py`

한 곳에서만 serialization을 정의해 학습과 평가의 형식이 달라지지 않게 한다.

기본 aligned sequence:

```text
###prompt
{prompt}

###chosen
{chosen}

###rejected
{rejected}
```

지원할 arrangement는 다음과 같다.

```text
aligned:       chosen 라벨에 chosen, rejected 라벨에 rejected
swapped:       chosen 라벨에 rejected, rejected 라벨에 chosen
```

지원할 segment order는 다음과 같다.

```text
forward:       chosen segment 후 rejected segment
reverse:       rejected segment 후 chosen segment
```

`swapped`는 학습 데이터에 사용하지 않고 평가용 diagnostic condition으로만 사용한다.

### `tokenization.py`

- OLMo-2 tokenizer 로드
- `input_ids`, `attention_mask` 생성
- prompt, delimiter, chosen answer, rejected answer의 token span 계산
- 각 답변의 target span을 표시하는 loss mask 생성
- 전체 sequence가 잘렸을 때 chosen/rejected target이 보존되는지 기록

단순히 전체 sequence loss를 계산하지 않고, 각 답변 span의 token loss를 별도로 집계한다.

### `training.py`

- `allenai/OLMo-2-0425-1B` base model 로드
- LoRA 설정 적용
- raw-text causal LM 학습
- checkpoint, trainer state, 학습 로그 저장

초기 학습은 일반 LoRA로 진행하고 QLoRA는 사용하지 않는다.

### `evaluation.py`

Base model과 LoRA model을 같은 evaluator로 평가한다.

각 sample에 대해 다음 조건을 생성한다.

| arrangement | order | 의미 |
|---|---|---|
| aligned | forward | 기본 serialization |
| aligned | reverse | 위치 순서 통제 |
| swapped | forward | 라벨과 답변 내용 불일치 |
| swapped | reverse | 불일치 + 위치 순서 통제 |

평가 결과에는 `scoring_mode`를 포함한다.

```text
full_sequence
isolated_branch
```

`full_sequence`에서는 실제 학습 serialization에서 각 답변 span의 loss를 계산한다. 이 경우 두 번째 답변은 첫 번째 답변까지 문맥으로 사용한다.

`isolated_branch`에서는 다음처럼 답변 하나만 포함한 sequence를 별도로 평가한다.

```text
###prompt
{prompt}

###{label}
{answer}
```

isolated branch는 라벨 효과와 답변 내용 효과를 해석하기에 더 깨끗한 보조 측정값으로 사용한다.

### `metrics.py`

- 답변별 token-level mean negative log-likelihood
- `chosen_loss`
- `rejected_loss`
- `loss_difference = chosen_loss - rejected_loss`
- chosen이 더 낮은 비율
- truncation 및 invalid sample 비율

sum loss는 답변 길이에 영향을 크게 받으므로 기본 지표로 사용하지 않는다. 다만 보조 지표로 보존할 수 있다.

## 4. 설정 파일 설계

`configs/sanity.yaml`은 모든 실행 설정을 한 곳에서 재현할 수 있게 한다.

```yaml
experiment_name: rawtext_dpo_sanity
seed: 42

dataset:
  name: allenai/Dolci-Instruct-DPO
  revision: aed155cf32e809b590490b6c3577ee4b0d0a5019
  split: train
  sample_size: 500
  eval_fraction: 0.1
  single_turn_only: true
  drop_empty_answers: true

model:
  name: allenai/OLMo-2-0425-1B
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
  num_train_epochs: 1
  max_seq_length: 1024
  serialization_order: forward

evaluation:
  arrangements: [aligned, swapped]
  orders: [forward, reverse]
  scoring_modes: [full_sequence, isolated_branch]

output:
  root_dir: outputs
```

본실험에서는 `sample_size`를 2,000~5,000으로 바꾸고 `max_seq_length`를 1,536 또는 2,048로 비교한다.

## 5. 데이터 분할 원칙

샘플링과 분할은 반드시 seed를 고정한다.

```text
전체 유효 단일 턴 데이터
    ↓ seed=42 sampling
500개 또는 2k~5k pair
    ↓ prompt_id 그룹 기준 분할
90% train / 10% evaluation
```

train과 evaluation에 같은 prompt가 들어가지 않도록 분할 후 prompt overlap 검사를 수행한다. `prompt_id`가 중복되거나 불완전하면 normalized prompt의 hash를 fallback key로 사용한다.

각 run에는 다음 manifest를 저장한다.

```text
run_id
dataset_name
dataset_revision
seed
sample_size
train_ids
eval_ids
filter_counts
```

## 6. 결과 파일 형식

### Sample-level 결과

`outputs/results/{run_id}/per_sample.csv`

```text
sample_id
prompt_id
model_variant
arrangement
order
scoring_mode
chosen_tokens
rejected_tokens
chosen_loss
rejected_loss
loss_difference
chosen_lower
truncated
```

### Aggregate 결과

`outputs/results/{run_id}/summary.csv`

```text
model_variant
arrangement
order
scoring_mode
n_samples
mean_chosen_loss
mean_rejected_loss
mean_loss_difference
median_loss_difference
chosen_lower_rate
```

### Run metadata

```text
config.yaml
dataset_manifest.json
environment.json
training_log.jsonl
```

## 7. 첫 실행 순서

```text
1. inspect_dataset.py
2. build_subset.py --config configs/sanity.yaml
3. 평가 코드로 Base / Original 조건 검증
4. Base / Swapped 및 Forward / Reverse 평가
5. train_lora.py 실행
6. 동일 네 조건을 LoRA model에 대해 평가
7. summarize_results.py로 CSV와 figure 생성
```

학습 전에 Base model에서 loss span과 Original/Swapped serialization이 기대한 문자열 위치에 매핑되는지 먼저 검증한다. 이 검증이 통과하지 않으면 LoRA 학습을 시작하지 않는다.

## 8. 설계상 주의점

1. 전체 raw sequence의 `rejected` loss는 앞의 chosen 답변에 조건부이다. 따라서 `full_sequence`와 `isolated_branch` 결과를 함께 보관한다.
2. `max_seq_length=1024`에서는 전체 sequence의 약 44%가 초과한다. 오른쪽 truncation은 rejected 답변에 더 큰 영향을 줄 수 있다.
3. `swapped` 조건은 학습 조건이 아니라 라벨과 내용의 관계를 확인하는 평가용 조건이다.
4. 순서 랜덤화는 token 수에는 거의 영향을 주지 않지만 autoregressive position effect를 바꾼다. 모든 evaluation sample을 forward와 reverse 양쪽으로 평가한다.
5. 전체 결과는 평균 하나로 끝내지 않고 sample-level CSV와 arrangement/order별 aggregate를 모두 저장한다.
