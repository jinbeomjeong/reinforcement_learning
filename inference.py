"""
학습된 DQN 모델로 CartPole 추론 및 시각화
==========================================
실행 전 train.py 학습으로 outputs/best_model.keras 가 생성되어 있어야 합니다.

실시간 화면:
  - pygame 창에 CartPole 제어 화면을 표시
  - 상단 HUD: 스텝 / 보상 / 에피소드 / 행동 방향
  - ESC 또는 창 닫기로 조기 종료 가능

출력물:
  outputs/inference_best.gif    - 최고 에피소드 애니메이션
  outputs/inference_states.png  - 최고 에피소드 상태 변수 그래프
  outputs/inference_summary.png - 전체 평가 에피소드 보상 요약
"""

import os
os.environ["KERAS_BACKEND"] = "torch"  # must be set before importing keras

import json
import time
from pathlib import Path
import numpy as np
import torch
import keras
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pygame

# ─────────────────────────────────────────
# 경로 / 상수
# ─────────────────────────────────────────
OUTPUT_DIR        = Path(__file__).resolve().parent / "outputs"
BEST_MODEL_PATH   = OUTPUT_DIR / "best_model.keras"
BEST_MODEL_METADATA_PATH = OUTPUT_DIR / "best_model_metadata.json"
GIF_PATH          = OUTPUT_DIR / "inference_best.gif"
STATES_PLOT_PATH  = OUTPUT_DIR / "inference_states.png"
SUMMARY_PLOT_PATH = OUTPUT_DIR / "inference_summary.png"

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_EPISODES   = 5     # 평가할 에피소드 수
GIF_DURATION_MS = 33    # GIF 프레임 간격 (ms) ≈ 30 fps
DISPLAY_FPS     = 50    # 실시간 창 목표 FPS
DISPLAY_SCALE   = 2     # 창 확대 배율 (600×400 → 1200×800)
PAUSE_ON_END    = 0.8   # 에피소드 종료 후 대기 시간 (초)

if matplotlib.get_backend().lower() == "agg":
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════
# 1. 모델 로드
# ══════════════════════════════════════════
def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first trainable parameter."""
    return next(model.parameters()).device


def load_model(path: Path) -> tuple[keras.Model, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            "먼저 train.py를 실행해 학습을 완료하세요."
        )
    model = keras.saving.load_model(path).to(DEVICE)
    model.eval()

    metadata = {}
    if BEST_MODEL_METADATA_PATH.exists():
        metadata = json.loads(BEST_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


# ══════════════════════════════════════════
# 2. HUD 렌더링 헬퍼
# ══════════════════════════════════════════
def _draw_hud(screen: pygame.Surface, font_lg: pygame.font.Font,
              font_sm: pygame.font.Font, step: int, reward: float,
              ep: int, total_eps: int, action: int, w: int) -> None:
    """화면 상단에 반투명 HUD를 그린다."""
    hud_h = 44
    hud = pygame.Surface((w, hud_h), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 170))
    screen.blit(hud, (0, 0))

    left_txt  = font_lg.render(f"Step {step:>3}   Reward {reward:>5.0f}", True, (220, 220, 220))
    right_txt = font_sm.render(
        f"Episode {ep}/{total_eps}   Action: {'LEFT <-' if action == 0 else '-> RIGHT'}",
        True, (160, 210, 255)
    )
    screen.blit(left_txt,  (12, 6))
    screen.blit(right_txt, (12, 26))


# ══════════════════════════════════════════
# 3. 실시간 에피소드 실행
#    - pygame 창에 실시간 표시
#    - rgb_array 프레임 동시 캡처 (GIF용)
# ══════════════════════════════════════════
def run_episode(
    env: gym.Env,
    model: keras.Model,
    screen: pygame.Surface,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    clock: pygame.time.Clock,
    seed: int = 0,
    ep_num: int = 1,
    total_eps: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, float, bool]:
    """
    반환값:
      frames      - rgb_array 프레임 리스트 (GIF용)
      states_arr  - (T, 4) 상태 배열
      reward      - 총 보상 (유지 스텝 수)
      aborted     - 사용자가 창을 닫았으면 True
    """
    state, _ = env.reset(seed=seed)

    W, H    = screen.get_width(), screen.get_height()
    env_w   = W // DISPLAY_SCALE
    env_h   = H // DISPLAY_SCALE

    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    total_reward = 0.0
    action       = 0
    aborted      = False

    for step in range(1000):
        # ── pygame 이벤트 처리 ──────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                aborted = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                aborted = True

        frame = env.render()          # (H_env, W_env, 3) uint8
        frames.append(frame.copy())
        states.append(state.copy())

        # numpy (H, W, C) → pygame surface → 확대
        surf = pygame.image.frombuffer(frame.tobytes(), (env_w, env_h), "RGB")
        surf_scaled = pygame.transform.scale(surf, (W, H))
        screen.blit(surf_scaled, (0, 0))
        _draw_hud(screen, font_lg, font_sm, step + 1, total_reward,
                  ep_num, total_eps, action, W)
        pygame.display.flip()
        clock.tick(DISPLAY_FPS)

        if aborted:
            break

        # ── 행동 선택 ───────────────────────────
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action  = int(model(state_t).argmax().item())

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            # 마지막 프레임 캡처
            frame = env.render()
            frames.append(frame.copy())
            states.append(state.copy())
            # 종료 상태 잠깐 표시
            surf = pygame.image.frombuffer(frame.tobytes(), (env_w, env_h), "RGB")
            screen.blit(pygame.transform.scale(surf, (W, H)), (0, 0))
            _draw_hud(screen, font_lg, font_sm, step + 2, total_reward,
                      ep_num, total_eps, action, W)
            pygame.display.flip()
            time.sleep(PAUSE_ON_END)
            break

    return frames, np.array(states), total_reward, aborted


# ══════════════════════════════════════════
# 4. GIF 저장
# ══════════════════════════════════════════
def save_gif(frames: list[np.ndarray], path: Path) -> None:
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    print(f"  [저장] GIF  → {path}  ({len(frames)} 프레임)")


# ══════════════════════════════════════════
# 5. 상태 변수 시각화 (최고 에피소드)
# ══════════════════════════════════════════
def plot_states(states: np.ndarray, reward: float, path: Path) -> None:
    steps  = range(len(states))
    labels = ["카트 위치 (m)", "카트 속도 (m/s)", "막대 각도 (rad)", "막대 각속도 (rad/s)"]
    colors = ["#4ecdc4", "#ff6b6b", "#ffd93d", "#c3a6ff"]
    limits = [2.4, None, 0.2095, None]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0f0f0f")

    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.plot(steps, states[:, i], color=colors[i], linewidth=1.6)
        ax.fill_between(steps, states[:, i], alpha=0.18, color=colors[i])
        ax.axhline(0, color="#555566", linewidth=0.8, linestyle="--")

        if limits[i] is not None:
            ax.axhline( limits[i], color="#ff4444", linewidth=1.0,
                        linestyle=":", label=f"한계 ±{limits[i]}")
            ax.axhline(-limits[i], color="#ff4444", linewidth=1.0, linestyle=":")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333355",
                      labelcolor="white", fontsize=9)

        ax.set_xlabel("스텝", color="#aaaaaa", fontsize=10)
        ax.set_ylabel(labels[i], color="#aaaaaa", fontsize=10)
        ax.set_title(labels[i], color="white", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"추론 상태 변화  |  총 {reward:.0f} 스텝 유지",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"  [저장] 상태 그래프 → {path}")


# ══════════════════════════════════════════
# 6. 전체 평가 요약 시각화
# ══════════════════════════════════════════
def plot_summary(rewards: list[float], best_ep: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#aaaaaa")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#333355")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ep_labels  = [f"EP {i+1}" for i in range(len(rewards))]
    bar_colors = ["#ffd93d" if i == best_ep else "#4ecdc4" for i in range(len(rewards))]
    bars = ax.bar(ep_labels, rewards, color=bar_colors, edgecolor="#222244", linewidth=0.6)

    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{r:.0f}", ha="center", va="bottom", color="white", fontsize=9)

    ax.axhline(500, color="#ff6b6b", linewidth=1.2, linestyle="--", label="최대 보상 (500)")
    ax.axhline(np.mean(rewards), color="#c3a6ff", linewidth=1.2,
               linestyle="-.", label=f"평균 {np.mean(rewards):.1f}")
    ax.set_ylim(0, 540)
    ax.set_xlabel("에피소드", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("보상 (스텝 수)", color="#aaaaaa", fontsize=11)
    ax.set_title("추론 평가 결과 요약  (노란색 = 최고 에피소드)",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"  [저장] 요약 그래프 → {path}")


# ══════════════════════════════════════════
# 7. 전체 평가 실행
# ══════════════════════════════════════════
def evaluate(model: keras.Model, n_episodes: int = EVAL_EPISODES) -> None:
    print(f"\n{'='*55}")
    print(f"  CartPole 추론 평가  ({n_episodes} 에피소드)")
    print(f"  ESC 또는 창 닫기로 현재 에피소드 중단 가능")
    print(f"{'='*55}")

    # ── pygame 초기화 ────────────────────────
    pygame.init()
    # CartPole 기본 렌더 크기는 600×400
    screen = pygame.display.set_mode((600 * DISPLAY_SCALE, 400 * DISPLAY_SCALE))
    pygame.display.set_caption("CartPole  DQN Inference")
    clock  = pygame.time.Clock()
    font_lg = pygame.font.SysFont(None, 30)
    font_sm = pygame.font.SysFont(None, 22)
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    rewards: list[float] = []
    best_reward  = -1.0
    best_ep_idx  = 0
    best_frames: list[np.ndarray] = []
    best_states: np.ndarray | None = None

    for ep in range(n_episodes):
        pygame.display.set_caption(
            f"CartPole  DQN Inference  |  Episode {ep+1}/{n_episodes}"
        )
        frames, states, reward, aborted = run_episode(
            env, model, screen, font_lg, font_sm, clock,
            seed=ep, ep_num=ep + 1, total_eps=n_episodes,
        )
        rewards.append(reward)

        tag = ""
        if reward > best_reward:
            best_reward = reward
            best_ep_idx = ep
            best_frames = frames
            best_states = states
            tag = "  ← 최고"
        print(f"  에피소드 {ep+1:2d} | 보상: {reward:5.0f} 스텝{tag}")

        if aborted:
            print("  [중단] 사용자가 창을 닫았습니다.")
            break

    env.close()
    pygame.quit()

    if not rewards:
        return

    print(f"\n  평균 보상 : {np.mean(rewards):.1f}")
    print(f"  최고 보상 : {best_reward:.0f}  (에피소드 {best_ep_idx+1})")

    # ── 파일 저장 ────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_gif(best_frames, GIF_PATH)
    plot_states(best_states, best_reward, STATES_PLOT_PATH)
    plot_summary(rewards, best_ep_idx, SUMMARY_PLOT_PATH)

    print(f"\n{'='*55}")
    print("  출력 파일")
    print(f"  - GIF        : {GIF_PATH}")
    print(f"  - 상태 그래프  : {STATES_PLOT_PATH}")
    print(f"  - 요약 그래프  : {SUMMARY_PLOT_PATH}")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════
# 실행
# ══════════════════════════════════════════
if __name__ == "__main__":
    print(f"  device : {DEVICE}")
    print(f"  모델 로드 : {BEST_MODEL_PATH}")

    model, metadata = load_model(BEST_MODEL_PATH)
    print(f"  model device : {model_device(model)}")
    if metadata:
        print(
            f"  학습 에피소드 : {metadata['episode']}  |  "
            f"학습 시 보상 : {metadata['reward']:.0f}"
        )

    evaluate(model, n_episodes=EVAL_EPISODES)
