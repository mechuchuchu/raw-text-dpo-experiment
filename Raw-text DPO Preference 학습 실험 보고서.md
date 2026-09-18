# Raw-text DPO Preference 학습 실험 보고서

## 1. 보고서 정보

- 작성 시각: 2026-09-18 17:53:05 UTC
- 실험 상태: 데이터셋 확보 및 구조·토큰 길이 사전 점검 완료
- 관련 계획서: `Raw-text DPO Preference 학습 실험 계획서.md`

## 2. 데이터셋

사용 데이터셋은 `allenai/Dolci-Instruct-DPO`이다.

- split: `train`
- 전체 preference pair: 259,922개
- 다운로드 크기: 약 810 MB
- materialized dataset 크기: 약 1.78 GB
- 로컬 Hugging Face 캐시: `/workspace/.hf_home`
- 현재 캐시 사용량: 약 2.5 GB

전체 데이터셋을 로컬 캐시에 다운로드하고 materialization까지 완료했다.

실제 컬럼은 다음과 같다.

```text
chosen
rejected
chosen_model
rejected_model
prompt_id
preference_type
```

계획서에서 가정한 별도 `prompt` 문자열 컬럼은 존재하지 않는다. `chosen`과 `rejected`가 각각 메시지 리스트이며, 일반적인 단일 턴 행은 다음 구조를 가진다.

```text
chosen:
  - role: user
    content: prompt
  - role: assistant
    content: chosen answer

rejected:
  - role: user
    content: 동일한 prompt
  - role: assistant
    content: rejected answer
```

## 3. 전체 데이터 구조 확인 결과

전체 259,922개 행을 기준으로 확인했다.

- 단일 턴 행: 249,923개
- 멀티턴 행: 9,999개
- 모든 행에서 chosen/rejected의 user 메시지는 동일했다.
- 대부분의 행은 `user → assistant` 구조였다.
- `rejected` assistant의 `content`가 `None`인 행: 41개
- 빈 rejected assistant 답변까지 포함한 유효하지 않은 행: 134개

`preference_type` 분포는 다음과 같다.

```text
delta_learning:              124,942
llm_judged:                   124,980
multiturn_self_talk:            5,000
multiturn_synthetic_context:    5,000
```

초기 실험에서는 멀티턴 처리와 결측 답변의 영향을 줄이기 위해 다음 조건의 행만 사용한다.

```text
chosen  = [user, assistant]
rejected = [user, assistant]
prompt, chosen answer, rejected answer가 모두 비어 있지 않은 문자열
```

전처리 결과는 다음처럼 만든다.

```text
prompt  = chosen[0]["content"]
chosen  = chosen[1]["content"]
rejected = rejected[1]["content"]
```

## 4. Token 길이 추정

`allenai/OLMo-2-0425-1B` tokenizer를 사용했다. seed 42로 섞은 유효 단일 턴 샘플 5,000개를 기준으로 추정했다.

전체 raw sequence는 다음 형식이다.

```text
###prompt
{prompt}

###chosen
{chosen}

###rejected
{rejected}
```

| 항목 | 최소 | 25% | 중앙값 | 75% | 90% | 95% | 99% | 최대 | 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prompt | 1 | 52 | 126 | 258 | 482 | 812 | 2,832 | 16,011 | 253.9 |
| Chosen answer | 1 | 118 | 347 | 799 | 1,424 | 1,913 | 3,409 | 15,663 | 595.5 |
| Rejected answer | 1 | 78 | 265 | 624 | 1,096 | 1,585 | 8,192 | 22,745 | 586.8 |
| Prompt + Chosen | 17 | 290 | 563 | 1,060 | 1,900 | 2,649 | 4,326 | 16,292 | 855.6 |
| Prompt + Rejected | 17 | 241 | 473 | 880 | 1,576 | 2,500 | 8,378 | 22,817 | 846.9 |
| 전체 raw sequence | 28 | 444 | 877 | 1,750 | 3,112 | 4,325 | 9,387 | 25,804 | 1,445.6 |

전체 raw sequence가 `max_seq_length`를 초과하는 비율은 다음과 같다.

```text
1024: 2,213 / 5,000 = 44.26%
1536: 1,463 / 5,000 = 29.26%
2048: 1,012 / 5,000 = 20.24%
```

## 5. 순서 랜덤화 확인

chosen/rejected의 의미와 라벨은 유지하고 sequence 내 segment 순서만 바꿔 비교했다.

```text
Forward:
###chosen C
###rejected R

Reverse:
###rejected R
###chosen C
```

5,000개 무작위 샘플에서 두 방식의 평균 길이는 모두 약 1,445.6 token이었다. 역순 길이에서 정순 길이를 뺀 평균은 0.01 token이었고, 3,996개 샘플은 전체 token 수가 완전히 같았다.

따라서 순서 랜덤화는 token 예산에는 거의 영향을 주지 않는다. 대신 autoregressive context에서 첫 번째 답변과 두 번째 답변의 위치 효과를 확인하는 평가 조건으로 사용할 수 있다.

## 6. 현재 실험 설정 결정

초기 sanity check는 다음과 같이 진행한다.

```text
데이터: 유효한 단일 턴 행
샘플 수: 500 pair
모델: allenai/OLMo-2-0425-1B
학습: LoRA
serialization: 계획서의 raw-text 형식
```

길이 설정은 다음을 기준으로 한다.

- `1024`: 빠른 코드 검증용. 전체 sequence 초과 행은 필터링하는 것이 안전하다.
- `1536`: 본실험 후보
- `2048`: 더 많은 답변을 보존하는 본실험 후보

오른쪽 truncation을 그대로 적용하면 sequence 뒤쪽에 있는 rejected 답변이 불리하게 잘릴 수 있다. 따라서 chosen/rejected loss를 비교할 때는 두 답변의 target token이 모두 보존되는 행을 사용하거나, 충분히 큰 sequence length를 사용해야 한다.

## 7. 다음 작업

1. 단일 턴 유효 행에서 고정 seed로 500개 sanity-check subset 생성
2. raw-text serialization 및 token mask 구현
3. Base 모델에서 Original, Swapped, 순서 Forward/Reverse 평가 구현
4. LoRA sanity 학습 수행
5. 동일한 조건에서 LoRA 모델 평가
6. sample별 chosen/rejected token loss와 길이를 CSV로 저장

평가 구현 시 전체 sequence에서 rejected loss를 계산하면 rejected 답변이 앞의 chosen 답변까지 문맥으로 사용한다는 점을 명시적으로 관리해야 한다. 답변 내용과 라벨 효과를 분리하기 위해 전체 raw sequence 평가와 독립 branch 평가를 구분하는 것이 좋다.
