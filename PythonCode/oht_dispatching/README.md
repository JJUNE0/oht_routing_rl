# OHT dispatching

OHT와 job의 할당·선택 로직을 담는 독립 패키지다.

- `oht_routing`: rail/path 비용과 경로 제어
- `oht_dispatching`: job-to-OHT 할당과 dispatch 정책

- `config.py`: dispatch mode와 선택 계약 버전
- `selector.py`: OHT 후보 생성, first-match/cost 선택, 진단 상태

runtime은 command 0에서 계산한 rail cost snapshot을 dispatcher에
전달하고, command 6에서는 simulator의 job/OHT 목록을 넘겨 할당 결과만
받는다.
