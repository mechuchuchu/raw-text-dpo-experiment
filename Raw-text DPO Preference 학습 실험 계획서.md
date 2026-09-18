# Raw-text DPO Preference 학습 실험

## 1. 연구 질문

DPO 데이터셋의 `prompt`, `chosen`, `rejected` 구조를 명시적인 preference learning loss 없이 하나의 raw text sequence로 만들어 causal language modeling 방식으로 학습했을 때, 모델이 `chosen`과 `rejected` 사이의 preference 정보를 학습하는가?

특히 학습 후 모델이 `###chosen`, `###rejected`라는 **라벨 자체**에 반응하는지, 아니면 답변의 **실제 내용**에 따른 likelihood 차이를 보이는지를 확인한다.

---

## 2. 기본 가설

DPO는 일반적인 language modeling과 달리 `chosen`과 `rejected`의 상대적인 확률을 직접 비교하여 학습한다.

반면 이번 실험에서는 다음과 같이 전체 데이터를 단순한 텍스트로 취급한다.

```text
###prompt
질문

###chosen
좋은 답변

###rejected
나쁜 답변
```

따라서 causal LM loss만 사용하면 모델은 기본적으로 `chosen`뿐만 아니라 `rejected`의 텍스트도 예측하도록 학습한다.

이 실험에서는 이러한 단순한 raw-text 학습만으로도 학습된 모델의 likelihood에 preference와 관련된 패턴이 나타나는지를 관찰한다.

---

## 3. 모델

### Base model

`allenai/OLMo-2-0425-1B`

약 1B parameter 규모의 OLMo 2 모델을 사용한다.

### Fine-tuning

LoRA를 사용한다.

초기 실험에서는 QLoRA 등의 추가적인 양자화 기법은 사용하지 않는다. 실험의 핵심 변수를 최소화하기 위해 모델은 가능한 한 일반적인 LoRA fine-tuning 환경으로 유지한다.

### Hardware

NVIDIA RTX 3060을 기준으로 한다.

초기 설정:

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 8~16
max_seq_length = 1024
```

GPU 메모리 상황에 따라 sequence length를 1536 또는 2048까지 늘릴 수 있다.

---

## 4. 데이터셋

`allenai/Dolci-Instruct-DPO`를 사용한다.

전체 데이터를 모두 학습시키지 않고 일부만 sampling하여 실험한다.

초기에는 다음 정도로 시작한다.

```text
전체 sample
    ↓
random sampling
    ↓
약 2,000~5,000 preference pairs
    ↓
90% train
10% evaluation
```

코드 및 학습 설정을 검증하기 위한 sanity check에서는 약 500개의 pair만 먼저 사용한다.

---

## 5. 데이터 전처리

원래의 preference pair를 다음과 같은 하나의 문자열로 변환한다.

```text
###prompt
{prompt}

###chosen
{chosen}

###rejected
{rejected}
```

이 문자열 전체를 causal language modeling의 하나의 training sequence로 취급한다.

즉, DPO loss나 reward model을 사용하지 않는다.

### 중요한 점

초기 실험에서는 `###prompt`, `###chosen`, `###rejected` 등의 delimiter도 그대로 입력에 포함한다.

또한 첫 실험에서는 전체 sequence에 일반적인 causal LM loss를 적용한다.

다만 평가에서는 prompt 및 delimiter의 loss와 답변 부분의 loss를 분리해서 측정한다.

---

## 6. Evaluation 설계

평가는 학습 전 Base model과 학습 후 LoRA model에서 각각 실시한다.

총 4개의 조건을 만든다.

### ① Base / Original

```text
###prompt
P

###chosen
C

###rejected
R
```

### ② Base / Swapped

```text
###prompt
P

###chosen
R

###rejected
C
```

여기서 중요한 것은 **라벨은 바꾸지 않는다는 것**이다.

즉 `###chosen`은 항상 `###chosen`으로 남기고, 그 안에 들어가는 답변만 교환한다.

### ③ LoRA / Original

```text
###prompt
P

###chosen
C

###rejected
R
```

### ④ LoRA / Swapped

```text
###prompt
P

###chosen
R

###rejected
C
```

---

## 7. Evaluation metric

전체 sequence loss 하나만 비교하지 않고 답변별 loss를 별도로 계산한다.

각 답변에 대해 token-level 평균 negative log-likelihood를 계산한다.

### Chosen loss

\[
L_C =
-\frac{1}{|C|}
\sum_t \log P(C_t | P, C_{<t})
\]

### Rejected loss

\[
L_R =
-\frac{1}{|R|}
\sum_t \log P(R_t | P, R_{<t})
\]

실제로는 각 답변이 원래 sequence에서 등장하는 위치를 기준으로 해당 부분의 token loss만 추출한다.

답변 길이가 서로 다를 수 있으므로 **sum loss가 아니라 평균 loss/token을 기본 metric으로 사용한다.**

---

## 8. 핵심 비교값

각 조건에서 다음 값을 기록한다.

```text
chosen_loss
rejected_loss
loss_difference = chosen_loss - rejected_loss
```

예를 들어:

```text
                 chosen loss    rejected loss
Base / Original      2.8            3.1
Base / Swapped       3.1            2.8

LoRA / Original      2.1            3.7
LoRA / Swapped       3.7            2.1
```

처럼 나타나는지를 관찰한다.

---

## 9. 결과 해석

특히 학습 전후의 변화와 swap에 대한 반응을 비교한다.

### 경우 A

학습 후에도 답변 내용에 따라 likelihood 차이가 유지된다.

```text
Original:
C loss < R loss

Swapped:
R loss < C loss
```

이 경우 모델이 단순히 `###chosen`이라는 문자열만 따라가는 것이 아니라, 답변 자체의 likelihood에도 영향을 받고 있을 가능성을 생각해볼 수 있다.

### 경우 B

`###chosen` 위치에 어떤 답변이 들어가느냐에 따라 loss가 크게 달라진다.

즉,

```text
###chosen
C
```

에서는 낮은 loss인데

```text
###chosen
R
```

에서는 높아지는 식의 패턴이 나타난다면, delimiter와 sequence 구조가 모델의 행동에 영향을 주고 있을 가능성이 있다.

### 경우 C

Original과 Swapped 사이의 차이가 거의 없다.

이 경우 raw-text LM 학습만으로는 `chosen/rejected` 구조가 뚜렷한 preference signal로 작동하지 않았을 가능성이 있다.

단, 이 결과만으로 "preference learning이 일어나지 않았다"고 단정하지 않고, 이번 실험의 범위에서는 해당 현상을 관찰했다고 표현한다.

---

## 10. 실험 순서

### Step 1 — Sanity check

약 500 pair만 사용한다.

학습 전에 다음이 정상적으로 계산되는지 확인한다.

```text
Base / Original
Base / Swapped
```

그리고 LoRA 학습 후:

```text
LoRA / Original
LoRA / Swapped
```

을 측정한다.

### Step 2 — 본실험

약 2,000~5,000 pair를 사용한다.

train/evaluation split은 학습 전에 고정한다.

```text
90% train
10% evaluation
```

가능하면 evaluation set의 prompt가 training set과 겹치지 않도록 한다.

### Step 3 — 결과 기록

각 조건에서 최소한 다음을 CSV 등으로 저장한다.

```text
sample_id
condition
chosen_loss
rejected_loss
loss_difference
```

그리고 전체 평균뿐만 아니라 sample별 결과도 보관한다.

---

## 11. 추가적으로 확인할 사항

실험 결과를 더 잘 해석하기 위해 evaluation set에서는 답변의 길이도 기록한다.

```text
chosen_tokens
rejected_tokens
```

왜냐하면 답변 길이가 크게 다르면 평균 loss 자체의 분포도 달라질 수 있기 때문이다.

또한 가능하면 `preference_type`, `chosen_model`, `rejected_model` 등의 Dolci-Instruct-DPO metadata도 보관한다.

본 실험에서는 필수 분석 대상으로 삼지는 않지만, 결과가 예상과 다를 경우 어떤 종류의 데이터에서 차이가 발생했는지 확인하는 데 사용할 수 있다.

---

## 12. 실험의 핵심 구조

전체 실험은 다음과 같이 정리된다.

```text
                 Dolci-Instruct-DPO
                         │
                    2k~5k pairs
                         │
                   90 / 10 split
                         │
              ┌──────────┴──────────┐
              │                     │
          Base model          LoRA training
              │                     │
        ┌─────┴─────┐               │
        │           │               │
     Original     Swapped       ┌────┴────┐
                                │         │
                             Original   Swapped

        ①           ②            ③         ④
```

각 조건에서:

```text
chosen answer loss
rejected answer loss
loss difference
```

를 측정한다.

---

## 13. 연구에서 가장 중요한 비교

최종적으로 가장 중요한 질문은 다음 두 가지다.

### 질문 1

Raw-text LM 학습 이후 `chosen`과 `rejected` 사이에 likelihood 차이가 나타나는가?

### 질문 2

그 차이가 `###chosen` / `###rejected`라는 **라벨 때문인지**, 아니면 답변 ​**내용 자체 때문인지**?

이를 확인하기 위해 Original과 Swapped evaluation을 비교한다.

---

## 14. 예상되는 프로젝트 규모

RTX 3060에서 무리하지 않는 작은 실험으로 구성한다.

```text
Model       : OLMo-2 1B
Fine-tuning : LoRA
Dataset     : Dolci-Instruct-DPO
Train size  : 약 2k~5k pairs
Eval size   : 약 200~500 pairs
Seq length  : 1024부터 시작
Evaluation  : 4 conditions
```

학습 시간이 오래 걸리는 것은 크게 문제가 되지 않으므로, GPU 메모리에 맞춰 batch size와 sequence length를 조절한다.

---

## 15. 프로젝트의 한계

이 실험은 DPO와 raw-text LM training을 엄밀하게 동일 조건에서 비교하는 연구라기보다는, **preference dataset을 일반적인 causal LM 데이터로 취급했을 때 어떤 현상이 나타나는지를 확인하는 탐색적 실험**이다.

특히 다음과 같은 요소가 결과에 영향을 줄 수 있다.

- `###chosen`, `###rejected` delimiter의 의미
- sequence 내에서의 위치
- chosen/rejected 답변의 길이
- Dolci-Instruct-DPO의 데이터 생성 방식
- OLMo-2 1B의 기존 학습 데이터와의 overlap
- random sampling에 따른 데이터 구성

따라서 결과에서는 "raw-text 학습이 DPO와 동일하다"는 식의 결론보다는, **raw-text serialization이 모델의 likelihood와 preference pair에 어떤 변화를 일으키는지**를 중심으로 해석한다.