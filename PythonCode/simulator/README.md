# Simulator integration

Simulator TCP protocol과 wire payload에 대응하는 데이터 모델을 관리한다.

- `client.py`: initialization/reset/active-data 송수신을 담당하는
  `PClient`
- `job.py`, `oht.py`, `rail_line.py`: simulator entity 모델
- 나머지 모듈: 통과 시간, 위치, command 시간 및 rail cost payload

이 패키지는 저수준 simulator 계약만 소유한다. 라우팅과 디스패칭 정책은
각각 `oht_routing`, `oht_dispatching`에서 이 패키지를 사용한다.
