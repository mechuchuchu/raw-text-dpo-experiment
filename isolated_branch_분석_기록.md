# Isolated-branch Preference 평가 기록

이 문서는 raw-text DPO 실험에서 `isolated_branch` 평가 결과를 기록하는 문서다.
`full_sequence`는 앞쪽 답변이 뒤쪽 답변의 문맥으로 사용되므로, 답변 자체에 대한 preference 비교는 이 문서를 기준으로 한다.

## 1. 평가 정의

각 응답을 다음과 같이 독립적으로 평가한다.

```text
prompt + label + answer
```

`loss_difference`는 다음과 같이 계산한다.

```text
loss_difference = chosen_loss - rejected_loss
```

- `loss_difference < 0`: chosen의 평균 token loss가 더 낮음
- `loss_difference > 0`: rejected의 평균 token loss가 더 낮음

`isolated_branch`에서는 답변 순서가 문맥에 영향을 주지 않으므로 `forward`와 `reverse` 결과가 동일하다.

## 2. 현재 실행 조건

| 항목 | 값 |
|---|---|
| 데이터셋 | `allenai/Dolci-Instruct-DPO` |
| 데이터 필터 | 유효한 단일 턴 preference pair |
| 모델 | `allenai/OLMo-2-0425-1B` |
| 학습 방식 | LoRA |
| 평가 방식 | `isolated_branch` |
| seed | 42로 생성된 subset 사용 추정 |
| Base 결과 파일 | `base_evaluation.csv` |
| LoRA 결과 파일 | `lora_evaluation.csv` |
| 비교 샘플 | Base와 LoRA에 공통으로 존재하는 346개 sample |
| 평가 행 수 | arrangement별 692행 (`forward`/`reverse` 중복 포함) |

> 현재 CSV에는 학습 설정의 모든 값이 포함되어 있지 않다. 다음 실행부터는 아래 재현성 기록 항목을 반드시 함께 저장한다.

## 3. 현재 결과

공통으로 평가된 346개 샘플을 기준으로 계산했다. `forward`와 `reverse`를 합친 행 기준이며, 두 순서의 값은 동일하다.

| arrangement | 모델 | chosen loss | rejected loss | loss difference | chosen 선호 비율 |
|---|---|---:|---:|---:|---:|
| aligned | Base | 1.5013 | 1.4518 | +0.0495 | 36.71% |
| aligned | LoRA | 1.1801 | 1.0945 | +0.0856 | 35.26% |
| swapped | Base | 1.5622 | 1.3581 | +0.2041 | 33.53% |
| swapped | LoRA | 1.2138 | 1.0948 | +0.1191 | 35.26% |

`chosen 선호 비율`은 sample별 `loss_difference < 0` 비율이다.

### LoRA 변화량

| arrangement | chosen loss 변화 | rejected loss 변화 | margin 변화 |
|---|---:|---:|---:|
| aligned | -0.3213 | -0.3573 | +0.0360 |
| swapped | -0.3483 | -0.2633 | -0.0850 |

## 4. 해석

1. LoRA 이후 chosen과 rejected의 절대 loss는 모두 감소했다. 따라서 모델이 해당 응답들의 표면적 likelihood를 더 잘 맞추게 된 효과는 있다.
2. 그러나 aligned 조건의 `loss_difference`는 `+0.0495`에서 `+0.0856`으로 증가했다. chosen을 rejected보다 더 선호하게 되었다고 보기 어렵다.
3. swapped 조건에서는 `+0.2041`에서 `+0.1191`로 차이가 줄었다.
4. swapped에서 `chosen`은 실제로 원래 rejected 응답이고, `rejected`는 원래 chosen 응답이다. 따라서 positive margin은 라벨을 뒤집어도 모델이 원래 chosen 응답을 더 낮은 loss로 평가한다는 뜻이다.
5. 현재 결과만으로는 LoRA가 preference 방향을 강화했다고 결론 내리기 어렵다. 절대 loss 감소와 preference margin 개선을 분리해서 봐야 한다.

## 5. 재실행 시 비교 규칙

full fine-tuning, 데이터셋 크기 변경, sequence length 변경 등을 비교할 때 다음 조건을 고정하거나 명시한다.

- 동일한 evaluation sample ID와 동일한 prompt group 분할
- 동일한 tokenizer와 serialization 형식
- 동일한 `max_seq_length`
- 동일한 truncation/filtering 정책
- 동일한 seed
- 동일한 평가 코드와 `scoring_mode=isolated_branch`
- Base와 학습 모델의 공통 sample만 사용한 paired comparison
- `chosen_loss`, `rejected_loss`, `loss_difference`, `loss_difference < 0` 비율을 모두 기록

데이터셋 크기나 학습 방식이 바뀌면 아래 표에 새 실행을 한 행으로 추가한다.

| run_id | 학습 방식 | train pair 수 | eval pair 수 | max seq length | epoch/step | lr | seed | aligned margin | swapped margin | aligned chosen 선호율 | swapped chosen 선호율 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current-lora | LoRA | 미기록 | 346 | 미기록 | 미기록 | 미기록 | 42 | +0.0495 → +0.0856 | +0.2041 → +0.1191 | 36.71% → 35.26% | 33.53% → 35.26% | 현재 CSV 기반 |

## 6. 다음 실행에서 추가할 메타데이터

최소한 다음 값을 결과 파일 또는 별도 `manifest.json`에 저장한다.

```text
run_id
model_name
training_method: lora | full_finetune | base
dataset_name
dataset_revision
train_pair_count
eval_pair_count
max_seq_length
learning_rate
num_train_epochs
max_steps
batch_size
gradient_accumulation_steps
lora_rank
lora_alpha
lora_dropout
seed
serialization_version
evaluation_script_version
```

## 7. 원본 결과 파일

- [base_evaluation.csv](base_evaluation.csv)
- [lora_evaluation.csv](lora_evaluation.csv)
- [실험 보고서](Raw-text%20DPO%20Preference%20학습%20실험%20보고서.md)
