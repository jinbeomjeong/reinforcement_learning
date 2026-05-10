# 프로젝트
mujoco 기반의 cartpole에 대해 강화학습으로 제어

## 가상환경
- miniconda를 사용
- miniconda의 설치 경로는 'C:\Users\{'로그인이름'}\miniconda3'
- 가상환경 이름은 'torch_210_gpu_python_311'

## mujoco 환경
- 설치 경로는 'C:\workspace\mujoco'

## 딥러닝 프레임워크
- pytorch를 백엔드로하는 keras

## 코딩 규칙
- 주어진 목적외에 추가적인 기능은 항상 질문하여 확인

## cartpole 환경
- 상태 공간: 위치, 속도, 각도, 각속도
- 행동 공간: 모터 지령(음수는 좌측이동, 양수는 우측 이동)

