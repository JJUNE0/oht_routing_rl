# OHT dispatching

OHT와 job의 할당·선택 로직을 담는 독립 패키지다.

- `oht_routing`: rail/path 비용과 경로 제어
- `oht_dispatching`: job-to-OHT 할당과 dispatch 정책

현재 contextual runtime의 command 6 선택 계약은
`oht_routing/routing/dispatch.py`에 유지한다. 디스패칭 구현을 정리할
때 이 패키지로 옮기며, 그 전까지는 양쪽 패키지 사이의 순환 import를
만들지 않는다.
