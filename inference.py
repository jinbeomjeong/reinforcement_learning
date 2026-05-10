import os
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

os.environ["KERAS_BACKEND"] = "torch"
import keras

if any(font.name == "Malgun Gothic" for font in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════ 설정 ══════════════════════════════════════
N_EPISODES       = 5            # 실행할 에피소드 수
MAX_STEPS        = 1000          # 에피소드 최대 스텝
INIT_CART_RANGE  = 0.5          # 초기 카트 위치 범위 [-0.5, 0.5] m
ANGLE_LIMIT_DEG  = 90         # 폴 각도 실패 한계 [-36, 36] deg
ANGLE_LIMIT = np.deg2rad(ANGLE_LIMIT_DEG)

INIT_ANGLE_DEG   = 30         # 초기 폴 각도 범위 [-17.2, 17.2] deg
INIT_ANGLE_RAD = np.deg2rad(INIT_ANGLE_DEG)

INIT_VEL_RANGE   = 0.05         # 초기 속도 범위 [-0.05, 0.05]
EPISODE_PAUSE    = 1.0          # 에피소드 종료 후 대기 시간 (초)
MODEL_ACTOR_PATH = "best_actor.weights.h5"
PLOT_SAVE_PATH   = "inference_result.png"

_DIR     = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_DIR, "cartpole.xml")

# train.py와 반드시 동일해야 함
ACTION_SCALE = 10.0
ACTION_DIM   = 1
N_STATES     = 4    # [카트위치, 카트속도, 폴각도, 폴각속도]

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CART_LIMIT  = 2.4
# ═════════════════════════════════════════════════════════════════════════════


def build_actor() -> keras.Model:
    # train.py의 SAC Actor 구조와 동일해야 가중치를 올바르게 로드할 수 있음
    return keras.Sequential([
        keras.layers.Input(shape=(N_STATES,)),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(ACTION_DIM * 2),   # 전반부: mean / 후반부: log_std
    ])


def load_actor(path: str) -> keras.Model:
    model = build_actor()
    model(torch.zeros(1, N_STATES))   # 가중치 초기화
    if DEVICE.type == "cuda":
        model.cuda()
    model.load_weights(path)
    print(f"[모델 로드] {path}")
    return model


def select_action(actor: keras.Model, state: np.ndarray) -> float:
    """결정론적 행동 선택 — 분포의 평균(mean)에 tanh 적용."""
    x = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out    = actor(x)                         # (1, ACTION_DIM * 2)
        mean   = out[:, :ACTION_DIM]              # 전반부만 사용
        action = torch.tanh(mean) * ACTION_SCALE
    return float(action.squeeze())


def get_state(data) -> np.ndarray:
    return data.sensordata.copy().astype(np.float32)


def is_terminated(data) -> bool:
    cart_pos   = float(data.qpos[0])
    pole_angle = float(data.qpos[1])
    return abs(cart_pos) > CART_LIMIT or abs(pole_angle) > ANGLE_LIMIT


def run_episode(mjmodel, data, actor, ep_num: int, viewer) -> tuple:
    mujoco.mj_resetData(mjmodel, data)
    # qpos[0]: 카트 위치 (slider), qpos[1]: 폴 각도 (hinge)
    data.qpos[0] = np.random.uniform(-INIT_CART_RANGE,  INIT_CART_RANGE)
    data.qpos[1] = np.random.uniform(-INIT_ANGLE_RAD, INIT_ANGLE_RAD)
    data.qvel[:] = np.random.uniform(-INIT_VEL_RANGE,   INIT_VEL_RANGE, mjmodel.nv)
    mujoco.mj_forward(mjmodel, data)

    total_reward   = 0.0
    step           = 0
    done           = False
    state_history  = []   # [(cart_pos, cart_vel, pole_angle, pole_vel), ...]
    action_history = []   # [motor_force (연속값), ...]
    last_action    = 0.0

    print(f"\n[에피소드 {ep_num:2d}] 시작")

    while not done and viewer.is_running():
        step_start = time.perf_counter()

        state  = get_state(data)
        action = select_action(actor, state)

        # 현재 상태·행동 기록 (mj_step 이전)
        state_history.append(state.copy())
        action_history.append(action)
        last_action = action

        data.ctrl[0] = action
        mujoco.mj_step(mjmodel, data)
        step += 1

        viewer.sync()

        state      = get_state(data)
        terminated = is_terminated(data)
        done       = terminated or (step >= MAX_STEPS)
        total_reward += 1.0 if not terminated else 0.0

        # 물리 타임스텝에 맞춰 실시간 페이싱
        elapsed   = time.perf_counter() - step_start
        remaining = mjmodel.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)

    result = "성공(최대스텝도달)" if step >= MAX_STEPS else "실패(폴 쓰러짐)"
    print(f"[에피소드 {ep_num:2d}] {result} | 보상={total_reward:.0f} | "
          f"스텝={step:3d} | 마지막제어력={last_action:+.2f} N")
    return total_reward, step, state_history, action_history


def save_inference_plots(all_states: list, all_actions: list, timestep: float) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    state_labels = ["카트 위치 (m)", "카트 속도 (m/s)", "폴 각도 (deg)", "폴 각속도 (deg/s)"]
    state_limits = [(-CART_LIMIT,  CART_LIMIT),  None,
                    (-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG),  None]

    for ep_idx, (states, actions) in enumerate(zip(all_states, all_actions)):
        sa       = np.array(states)              # (T, 4)
        plot_sa  = sa.copy()
        plot_sa[:, 2] = np.rad2deg(plot_sa[:, 2])
        plot_sa[:, 3] = np.rad2deg(plot_sa[:, 3])
        t        = np.arange(len(sa)) * timestep # 시간 축 (초)
        c        = colors[ep_idx % len(colors)]
        lbl      = f"에피소드 {ep_idx + 1}"

        # ── 상태 공간 시계열 ────────────────────────────────────────────────
        axes[0, 0].plot(t, plot_sa[:, 0], color=c, alpha=0.8, label=lbl)
        axes[0, 1].plot(t, plot_sa[:, 1], color=c, alpha=0.8, label=lbl)
        axes[1, 0].plot(t, plot_sa[:, 2], color=c, alpha=0.8, label=lbl)
        axes[1, 1].plot(t, plot_sa[:, 3], color=c, alpha=0.8, label=lbl)

        # ── 행동 공간 시계열 (연속값 → 일반 line plot) ─────────────────────
        axes[2, 0].plot(t, actions, color=c, alpha=0.8, label=lbl)

        # ── 위상 궤적: 폴 각도 vs 폴 각속도 ────────────────────────────────
        axes[2, 1].plot(plot_sa[:, 2], plot_sa[:, 3], color=c, alpha=0.7, label=lbl)
        axes[2, 1].plot(plot_sa[0, 2], plot_sa[0, 3], "o", color=c, markersize=5)   # 시작점

    # ── 축 설정 ─────────────────────────────────────────────────────────────
    time_label = "시간 (s)"
    axes[0, 0].set(title="카트 위치",         xlabel=time_label, ylabel=state_labels[0])
    axes[0, 1].set(title="카트 속도",         xlabel=time_label, ylabel=state_labels[1])
    axes[1, 0].set(title="폴 각도",           xlabel=time_label, ylabel=state_labels[2])
    axes[1, 1].set(title="폴 각속도",         xlabel=time_label, ylabel=state_labels[3])
    axes[2, 0].set(title="행동 (연속 제어력)", xlabel=time_label, ylabel="제어력 (N)")
    axes[2, 1].set(title="위상 궤적 (폴)",    xlabel=state_labels[2], ylabel=state_labels[3])

    # 종료 한계선
    for ax, lim in zip(axes.flat[:4], state_limits):
        if lim:
            ax.axhline(lim[0], color="red", linestyle="--", alpha=0.4, linewidth=1)
            ax.axhline(lim[1], color="red", linestyle="--", alpha=0.4, linewidth=1,
                       label="종료 한계")
    axes[2, 0].axhline(0, color="k", linestyle="--", alpha=0.3)
    axes[2, 0].set_ylim(-ACTION_SCALE * 1.1, ACTION_SCALE * 1.1)
    axes[2, 1].axvline(-ANGLE_LIMIT_DEG, color="red", linestyle="--", alpha=0.4)
    axes[2, 1].axvline( ANGLE_LIMIT_DEG, color="red", linestyle="--", alpha=0.4,
                        label="종료 한계")
    axes[2, 1].axhline(0, color="k", linestyle="--", alpha=0.3)
    axes[2, 1].axvline(0, color="k", linestyle="--", alpha=0.3)

    for ax in axes.flat:
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(fig)
    print(f"[저장] 추론 결과 그래프 → {PLOT_SAVE_PATH}")


def main():
    print(f"[장치] {DEVICE}")
    print(f"[알고리즘] SAC (결정론적 추론 — 분포 평균 사용)")

    if not os.path.exists(MODEL_ACTOR_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_ACTOR_PATH}")
        print("  먼저 train.py를 실행하여 모델을 학습하세요.")
        return

    actor   = load_actor(MODEL_ACTOR_PATH)
    mjmodel = mujoco.MjModel.from_xml_path(XML_PATH)
    data    = mujoco.MjData(mjmodel)
    results     = []
    all_states  = []
    all_actions = []

    print(f"[추론] {N_EPISODES}개 에피소드 실행 (뷰어 창 닫기로 종료)\n")

    with mujoco.viewer.launch_passive(mjmodel, data) as viewer:
        for ep in range(1, N_EPISODES + 1):
            if not viewer.is_running():
                break

            reward, steps, states, actions = run_episode(mjmodel, data, actor, ep, viewer)
            results.append((reward, steps))
            all_states.append(states)
            all_actions.append(actions)

            # 에피소드 종료 후 잠시 현재 상태 유지 (결과 확인용)
            pause_end = time.perf_counter() + EPISODE_PAUSE
            while time.perf_counter() < pause_end and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

    if results:
        rewards, steps_list = zip(*results)
        sep = "=" * 45
        print(f"\n{sep}")
        print(f"[최종 결과] {len(results)}개 에피소드")
        print(f"  평균 보상 : {np.mean(rewards):.1f}")
        print(f"  최대 보상 : {np.max(rewards):.0f}")
        print(f"  성공 횟수 : {sum(s >= MAX_STEPS for s in steps_list)}/{len(results)}")
        print(f"  평균 스텝 : {np.mean(steps_list):.1f}")
        print(sep)

        save_inference_plots(all_states, all_actions, timestep=mjmodel.opt.timestep)


if __name__ == "__main__":
    main()
