# Project TODO

이 문서는 이 프로젝트를 `LLM/VLM Engineer, ADAS` 포지션 관점에서 정리한 작업 메모다.
목표는 "예쁘게 보이는 데모"보다 `데이터 생성 파이프라인`, `평가`, `개선 루프`를 명확히 만드는 것이다.

## Current Goal

자연어 요청을 SUMO 시뮬레이션 파라미터/구조로 바꾸고, 사람 수정 로그를 축적해 재학습과 평가로 닫히는 루프를 만든다.

핵심 2축:
- `LLM-level evaluation`: 자연어 -> SimulationParams 품질 평가
- `System-level evaluation`: SimulationParams -> SUMO 결과 재현도 평가

## Done

- FT 모델과 베이스 LLM 역할 분리
- 웹 생성 페이지 구축
- 수정 모드 안정화
- `오류 수정(correction)` / `대안 요청(alternative)` 분리
- 수정 로그 구조화 저장
- correction-only export 구현
- 관리자 페이지 `/admin` 추가
- evaluation report 다운로드 추가
- SUMO 워밍업 구간 도입

## High Priority

### 1. LLM-level evaluation 만들기

목표:
- 모델이 `자연어 -> 파라미터`를 얼마나 잘 뽑는지 평가

우선 구현:
- correction 로그 기반 평가 리포트
- 초기 파라미터 vs 오류 수정 후 파라미터 차이 분석
- 필드별 수정 빈도 집계
- 필드별 평균 수정량 집계

보고 싶은 지표:
- correction rate
- `vehicles_per_hour` 수정 빈도
- `speed_limit_kmh` 수정 빈도
- `sigma`, `tau`, `lanes` 수정 빈도
- geometry vs parameter 오류 비율

### 2. Dataset replay evaluation 만들기

목표:
- 기존 학습/생성 데이터셋을 다시 모델에 넣어 오프라인 평가

할 일:
- 평가용 prompt set 파일 정의
- expected params 혹은 rule-based expected range 정의
- base model vs fine-tuned model 비교

예시 지표:
- location extraction accuracy
- lanes exact match
- speed limit range pass rate
- volume range pass rate
- domain consistency pass rate

### 3. 관리자 페이지 확장

목표:
- 운영/실험 관리용 UI 완성

할 일:
- correction 로그 기반 LLM evaluation 카드 추가
- 필드별 error pattern 표 추가
- 다운로드 가능한 JSON/TXT report 추가
- 시뮬레이션 상세 보기 추가

## Medium Priority

### 4. SUMO 파라미터 반영도 개선

문제:
- 현재 `volume`, `lanes`, `speed_limit`은 비교적 잘 반영되지만
- `sigma`, `tau`는 SUMO vehicle behavior에 완전하게 연결되지 않음

할 일:
- FT가 뽑은 `sigma`, `tau`를 vType에 동적으로 반영
- 결과 속도/대기시간 변화 확인

### 5. Error taxonomy 정리

목표:
- 어떤 오류가 많은지 더 구체적으로 분류

후보:
- location hallucination
- speed overestimate
- speed underestimate
- geometry mismatch
- parameter mismatch
- needs correction
- unverified

### 6. GCP 배포 구조 정리

목표:
- 나중에 Cloud Run / Job으로 올릴 수 있게 정리

할 일:
- 웹 서버와 SUMO 실행 분리 여부 결정
- 다운로드/report 경로 클라우드 환경 기준으로 점검
- headless SUMO 기준 배포 문서 작성

## Low Priority

### 7. 편집 UI 고도화

주의:
- 이건 JD 관점에서 우선순위가 낮다
- 풀 편집기보다 구조/평가/로그가 더 중요하다

정말 필요할 때만:
- 선택형 도로 편집
- edge 선택 후 geometry 조정

## Next Recommended Step

다음 바로 할 것:
- `correction 로그 기반 LLM evaluation report` 구현

구체 작업:
1. 초기 파라미터와 수정 후 파라미터 diff 집계
2. 필드별 correction 빈도 계산
3. 관리자 페이지에 LLM evaluation 섹션 추가
4. report 다운로드 추가

## Notes

- `alternative`는 학습 데이터에 넣지 않는다.
- `correction`만 trainable 데이터로 사용한다.
- JD 관점에서 중요한 건 UI 화려함보다:
  - prompt / FT 개선
  - dataset generation pipeline
  - evaluation
  - error analysis
  - improvement cycle
