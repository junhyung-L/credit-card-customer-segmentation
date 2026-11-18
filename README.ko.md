# 신용카드 고객 세그먼트 분류

고객·신용·매출·청구·잔액·채널·마케팅 월별 컬럼 857개로 신용카드 고객을 다섯 세그먼트(A–E)로 나누는 다중 분류 모델입니다. DACON 신용카드 고객 세그먼트 분류 AI 경진대회에 참가하며 만들었습니다.

**문제.** 세그먼트는 고객에게 어떤 카드 상품과 캠페인을 제안할지 정하지만, 클래스 분포가 극단적으로 치우쳐 있습니다. 학습 데이터 2,400,000행 중 E가 80.1%이고 A와 B는 합쳐도 0.05%가 안 됩니다. 상당수 컬럼은 대부분 비어 있고, 비어 있는 이유도 테이블마다 다릅니다.

**결과.** DACON private 리더보드 점수 0.6251(58위, 상위 25%). 내부 2만 건 표본에서는 CatBoost + LR + MLP 스태킹이 weighted F1 0.8936을 기록했습니다. 항상 E로 예측하면 0.7026입니다.

[English](README.md) · [사례 연구](PORTFOLIO.ko.md) · [실험 요약](reports/실험요약.md)

## 데이터

월별 고객 기록 기준 학습 2,400,000행, 테스트 600,000행입니다. 월별 도메인 테이블을 합치면 857개 컬럼이고, 타깃은 다중 클래스 `Segment`입니다. 학습 세그먼트 분포는 E 1,922,052, D 349,242, C 127,590, A 972, B 144입니다. 원본 데이터는 저장소에 없고, `data/data`에 대회 데이터 페이지 링크가 있습니다.

## 접근 방식

```mermaid
flowchart LR
    A[월별 원천 테이블\n저장소 미포함] --> B[월·ID 기준 결합]
    B --> C[결측 원인 분석과 규칙 기반 전처리]
    C --> D[인코딩과 스케일링]
    D --> E[표본 기반 모델 비교]
    E --> F[XGBoost·CatBoost·신경망\nTabNet·스태킹]
    F --> G[weighted F1 검증]
    G --> H[대회 제출]
```

- **데이터 결합:** `src/baseline_xgb.py`가 월별 테이블을 도메인별로 이어 붙이고, 월과 `ID` 기준으로 left merge합니다.
- **결측 분석:** `src/missing_mechanism_analysis.py`가 결측률을 계산하고, 규칙 기반 채움과 multi-output random forest 대체를 시험합니다.
- **피처 준비:** 결측률 50% 이상 컬럼은 버리고, 20~50%는 모델 기반으로 채우며, 20% 미만은 `Unknown` 범주로 둡니다. 로그 변환과 표준화를 적용합니다.
- **모델 비교:** 2만 건 표본에서 CatBoost, 로지스틱 회귀(LR), XGBoost, 랜덤 포레스트, DNN/MLP/CNN(심층·다층·합성곱 신경망), TabNet, 여러 스태킹 구성을 비교합니다.

설계 선택과 버린 대안: [PORTFOLIO.ko.md](PORTFOLIO.ko.md#주요-결정).

## 결과

**DACON 리더보드**

| 제출 | Public | Private |
|---|---:|---:|
| 기준 XGBoost | 0.607 | – |
| 결측 패턴 반영 XGBoost | 0.625 | – |
| 최종 제출 | 0.6464(75위) | **0.6251**(58위, 상위 25%) |

**내부 2만 건 표본(weighted F1)**

| 모델 | weighted F1 |
|---|---:|
| 항상 E 예측(참고) | 0.7026 |
| TabNet | 0.8285 |
| CatBoost | 0.8893 |
| CatBoost + LR + MLP 스태킹(LR 메타 모델) | **0.8936** |

내부 값은 모두 학습 데이터에서 뽑은 같은 20,000행 평가 표본에서 계산했습니다. 두 표는 데이터와 채점 방식이 달라 숫자를 서로 비교할 수 없습니다.

![모델 실험별 weighted F1 비교](images/model_f1_comparison.png)

*그림. 2만 건 평가 표본의 weighted F1입니다. 축이 0.82에서 시작해 차이가 실제보다 커 보이며, 다수 세그먼트 기준값(0.7026)은 그림 왼쪽 밖에 있습니다.*

## 실행

```powershell
pip install -r requirements.txt
python src\preprocess_missing_features.py
```

대회 파일을 [research/RUN_MANIFEST.md](research/RUN_MANIFEST.md) 설명대로 두면 결측 패턴을 반영한 기본 CSV가 만들어집니다(`--drop-threshold` 기본값 0.50 이상인 컬럼은 제거). 2만 건 모델 비교는 출력과 함께 `notebooks/train_eval_20k.ipynb`에 있습니다.

## 한계

- 내부 weighted F1(0.8936)과 리더보드 점수(0.6251)는 데이터와 채점 방식이 다릅니다. 외부 기준은 리더보드 점수뿐입니다.
- 표본을 고객이 아닌 월별 행 단위로 뽑아, 같은 고객이 학습 표본과 평가 표본에 함께 들어갈 수 있습니다.
- weighted F1은 E 세그먼트가 좌우합니다. 평가 표본에는 A가 11행, B는 0행이며, 클래스별 F1은 계산하지 않았습니다.
- CatBoost 대비 스태킹의 개선 폭(+0.0043)은 반복 추출 없이 표본 하나에서 나온 값입니다.

## 저장소 구성

```text
.
├── src/                         # 내보낸 Colab 전처리·학습 스크립트
├── notebooks/                   # 출력이 저장된 노트북(결과의 출처)
├── reports/실험요약.md           # 실험 요약과 점수
├── images/model_f1_comparison.png
└── data/                        # 데이터 출처 링크만 있는 자리표시자
```

## 문서

- [포트폴리오 사례 연구](PORTFOLIO.ko.md)
- [아키텍처](docs/ARCHITECTURE.md)
- [실행 매니페스트](research/RUN_MANIFEST.md)
