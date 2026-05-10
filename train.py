import os
import random
import collections
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn.functional as F
from torch.distributions import Normal

os.environ["KERAS_BACKEND"] = "torch"
import keras

if any(font.name == "Malgun Gothic" for font in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════ 설정 ══════════════════════════════════════
# ── 실행 설정 ────────────────────────────────────────────────────────────────
N_EPISODES     = 1000      # 총 학습 에피소드 수
PRINT_INTERVAL = 50        # 상태 출력 에피소드 간격
MAX_STEPS      = 500       # 에피소드 최대 스텝

# ── 리플레이 메모리 / 배치 ──────────────────────────────────────────────────
BATCH_SIZE  = 64           # 미니배치 크기
MEMORY_SIZE = 10_000       # 리플레이 메모리 크기

# ── SAC 학습 하이퍼파라미터 ─────────────────────────────────────────────────
GAMMA          = 0.99      # 할인율
ACTOR_LR       = 3e-4      # Actor 학습률
CRITIC_LR      = 3e-4      # Critic 학습률
ALPHA_LR       = 3e-4      # 온도 파라미터(α) 학습률
TAU            = 0.005     # 소프트 타겟 업데이트 계수
LOG_ALPHA_INIT = 0.0       # 초기 온도: α = exp(0) = 1.0

# ── Actor 출력 / 행동 공간 ──────────────────────────────────────────────────
ACTION_SCALE   = 10.0      # 모터 제어력 범위 [-10, 10]
ACTION_DIM     = 1         # 연속 행동 차원
LOG_STD_MIN    = -20       # log_std 하한
LOG_STD_MAX    = 2         # log_std 상한
TARGET_ENTROPY = -float(ACTION_DIM)  # 목표 엔트로피 (휴리스틱: -dim(A))

# ── 환경 종료 조건 ──────────────────────────────────────────────────────────
SOFT_CART_LIMIT = 1.5      # 카트 위치 제어 실패 한계 [-1.5, 1.5] m
ANGLE_LIMIT_DEG = 90       # 폴 각도 실패 한계 [-90, 90] deg
ANGLE_LIMIT_RAD = np.deg2rad(ANGLE_LIMIT_DEG)

# ── 초기 상태 랜덤화 ────────────────────────────────────────────────────────
INIT_CART_RANGE = 0.5      # 초기 카트 위치 범위 [-0.5, 0.5] m
INIT_ANGLE_DEG  = 45       # 초기 폴 각도 범위 [-45, 45] deg
INIT_ANGLE_RAD  = np.deg2rad(INIT_ANGLE_DEG)
INIT_VEL_RANGE  = 0.1      # 초기 속도 범위 [-0.1, 0.1]

# ── 저장 경로 ───────────────────────────────────────────────────────────────
MODEL_ACTOR_PATH   = "best_actor.weights.h5"
MODEL_CRITIC1_PATH = "best_critic1.weights.h5"
MODEL_CRITIC2_PATH = "best_critic2.weights.h5"
PLOT_SAVE_PATH     = "training_result.png"

# ── 파생 설정 / 런타임 설정 ─────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_DIR, "cartpole.xml")

N_STATES        = 4   # [카트위치, 카트속도, 폴각도, 폴각속도]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ═════════════════════════════════════════════════════════════════════════════


class ReplayMemory:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def put(self, state, action: float, reward: float, next_state, done_mask: float):
        self.buffer.append((
            np.asarray(state,      dtype=np.float32),
            np.float32(action),
            np.float32(reward),
            np.asarray(next_state, dtype=np.float32),
            np.float32(done_mask),
        ))

    def sample(self, n: int):
        batch = random.sample(self.buffer, n)
        s, a, r, ns, dm = zip(*batch)
        _t = lambda x, dt: torch.tensor(np.array(x), dtype=dt).to(DEVICE)
        return (
            _t(s,  torch.float32),
            _t(a,  torch.float32).unsqueeze(1),
            _t(r,  torch.float32).unsqueeze(1),
            _t(ns, torch.float32),
            _t(dm, torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class CartPoleEnv:
    _CART_LIMIT  = 2.4
    _ANGLE_LIMIT = ANGLE_LIMIT_RAD

    def __init__(self, xml_path: str, max_steps: int = MAX_STEPS):
        self.model     = mujoco.MjModel.from_xml_path(xml_path)
        self.data      = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self._steps    = 0

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        # qpos[0]: 카트 위치 (slider), qpos[1]: 폴 각도 (hinge)
        self.data.qpos[0] = np.random.uniform(-INIT_CART_RANGE,  INIT_CART_RANGE)
        self.data.qpos[1] = np.random.uniform(-INIT_ANGLE_RAD, INIT_ANGLE_RAD)
        self.data.qvel[:] = np.random.uniform(-INIT_VEL_RANGE,   INIT_VEL_RANGE, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_state()

    def step(self, action: float):
        self.data.ctrl[0] = float(np.clip(action, -ACTION_SCALE, ACTION_SCALE))
        mujoco.mj_step(self.model, self.data)
        self._steps += 1

        state      = self._get_state()
        cart_pos   = float(self.data.qpos[0])
        pole_angle = float(self.data.qpos[1])
        terminated = (abs(cart_pos) > SOFT_CART_LIMIT or
                      abs(pole_angle) > self._ANGLE_LIMIT)
        done       = terminated or (self._steps >= self.max_steps)
        if terminated:
            reward = 0.0
        else:
            position_penalty = (cart_pos / self._CART_LIMIT) ** 2
            reward = 1.0 - position_penalty
        return state, reward, done

    def _get_state(self) -> np.ndarray:
        # sensordata 순서: [카트위치, 카트속도, 폴각도, 폴각속도]
        return self.data.sensordata.copy().astype(np.float32)


# ── Actor: 상태 → (mean, log_std) ─────────────────────────────────────────────
# Sequential 단일 출력(ACTION_DIM * 2)을 sample_action에서 분리
def build_actor() -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(shape=(N_STATES,)),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(ACTION_DIM * 2),   # 전반부: mean / 후반부: log_std
    ])


# ── Critic: concat(상태, 행동) → Q값 ──────────────────────────────────────────
def build_critic() -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(shape=(N_STATES + ACTION_DIM,)),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.Dense(1),
    ])


def init_networks():
    actor          = build_actor()
    critic1        = build_critic()
    critic2        = build_critic()
    critic1_target = build_critic()
    critic2_target = build_critic()

    dummy_s  = torch.zeros(1, N_STATES)
    dummy_sa = torch.zeros(1, N_STATES + ACTION_DIM)
    actor(dummy_s)
    critic1(dummy_sa);        critic2(dummy_sa)
    critic1_target(dummy_sa); critic2_target(dummy_sa)

    if DEVICE.type == "cuda":
        actor.cuda()
        critic1.cuda();        critic2.cuda()
        critic1_target.cuda(); critic2_target.cuda()

    critic1_target.set_weights(critic1.get_weights())
    critic2_target.set_weights(critic2.get_weights())

    log_alpha = torch.tensor(
        [LOG_ALPHA_INIT], dtype=torch.float32, requires_grad=True, device=DEVICE
    )
    return actor, critic1, critic2, critic1_target, critic2_target, log_alpha


def soft_update(src: keras.Model, tgt: keras.Model) -> None:
    src_weights = src.get_weights()
    tgt_weights = tgt.get_weights()
    tgt.set_weights([
        TAU * src_w + (1.0 - TAU) * tgt_w
        for src_w, tgt_w in zip(src_weights, tgt_weights)
    ])


def sample_action(actor: keras.Model, states: torch.Tensor, deterministic: bool = False):
    """
    재매개변수화 트릭으로 행동 샘플링.
    반환: (action, log_prob)  — deterministic=True 시 log_prob=None
    """
    out     = actor(states)                              # (B, ACTION_DIM * 2)
    mean    = out[:, :ACTION_DIM]
    log_std = out[:, ACTION_DIM:].clamp(LOG_STD_MIN, LOG_STD_MAX)
    std     = log_std.exp()

    if deterministic:
        action = torch.tanh(mean) * ACTION_SCALE
        return action, None

    dist  = Normal(mean, std)
    x_t   = dist.rsample()                              # 재매개변수화 샘플
    y_t   = torch.tanh(x_t)
    action = y_t * ACTION_SCALE

    # tanh squashing 보정이 포함된 로그 확률
    log_prob = dist.log_prob(x_t) - torch.log(ACTION_SCALE * (1 - y_t.pow(2)) + 1e-6)
    log_prob = log_prob.sum(dim=1, keepdim=True)
    return action, log_prob


def select_action(actor: keras.Model, state: np.ndarray) -> float:
    """학습 중 환경 상호작용 시 확률론적으로 행동 선택."""
    x = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        action, _ = sample_action(actor, x, deterministic=False)
    return float(action.squeeze())


def train_step(actor, critic1, critic2, critic1_target, critic2_target,
               memory, actor_opt, critic1_opt, critic2_opt, log_alpha, alpha_opt):
    states, actions, rewards, next_states, done_masks = memory.sample(BATCH_SIZE)
    alpha = log_alpha.exp().detach()

    # ── Critic 업데이트 ─────────────────────────────────────────────────────
    with torch.no_grad():
        next_actions, next_log_probs = sample_action(actor, next_states)
        sa_next = torch.cat([next_states, next_actions], dim=1)
        q_next  = torch.min(critic1_target(sa_next),
                            critic2_target(sa_next)) - alpha * next_log_probs
        target  = rewards + GAMMA * q_next * done_masks

    sa      = torch.cat([states, actions], dim=1)
    c1_loss = F.huber_loss(critic1(sa), target)
    c2_loss = F.huber_loss(critic2(sa), target)

    critic1_opt.zero_grad(); c1_loss.backward(); critic1_opt.step()
    critic2_opt.zero_grad(); c2_loss.backward(); critic2_opt.step()

    # ── Actor 업데이트 (critic 가중치 고정: actor gradient만 계산) ───────────
    for p in list(critic1.parameters()) + list(critic2.parameters()):
        p.requires_grad = False

    pi, log_pi = sample_action(actor, states)
    sa_pi      = torch.cat([states, pi], dim=1)
    actor_loss = (alpha * log_pi - torch.min(critic1(sa_pi), critic2(sa_pi))).mean()

    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    for p in list(critic1.parameters()) + list(critic2.parameters()):
        p.requires_grad = True

    # ── 온도 파라미터(α) 업데이트 ────────────────────────────────────────────
    alpha_loss = -(log_alpha * (log_pi.detach() + TARGET_ENTROPY)).mean()
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    # ── 소프트 타겟 업데이트 ────────────────────────────────────────────────
    soft_update(critic1, critic1_target)
    soft_update(critic2, critic2_target)

    return (c1_loss.item() + c2_loss.item()) / 2, actor_loss.item(), log_alpha.exp().item()


def save_plots(rewards: list, critic_losses: list, actor_losses: list,
               alphas: list) -> None:
    eps = np.arange(1, len(rewards) + 1)
    win = 50

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes

    def plot_with_ma(ax, data, color, label, title, ylabel):
        ax.plot(eps, data, alpha=0.35, color=color)
        if len(data) >= win:
            ma = np.convolve(data, np.ones(win) / win, mode="valid")
            ax.plot(eps[win - 1:], ma, color=color, label=f"{label} ({win}ep MA)")
        ax.set(title=title, xlabel="에피소드", ylabel=ylabel)
        ax.legend(); ax.grid(alpha=0.3)

    plot_with_ma(ax1, rewards,       "steelblue", "보상",       "학습 보상 추이",        "누적 보상")
    plot_with_ma(ax2, critic_losses, "tomato",    "Critic 손실", "Critic 손실 추이",      "Huber 손실")
    plot_with_ma(ax3, actor_losses,  "seagreen",  "Actor 손실",  "Actor 손실 추이",       "손실 (α·H - Q)")
    plot_with_ma(ax4, alphas,        "orchid",    "α",           "온도 파라미터(α) 추이", "α 값")

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(fig)
    print(f"[저장] 학습 결과 그래프 → {PLOT_SAVE_PATH}")


def main():
    print(f"[장치] {DEVICE}")
    print(f"[알고리즘] SAC (Soft Actor-Critic)")
    print(f"[설정] 에피소드={N_EPISODES} | 배치={BATCH_SIZE} | γ={GAMMA} | τ={TAU}")
    print(f"       actor_lr={ACTOR_LR} | critic_lr={CRITIC_LR} | alpha_lr={ALPHA_LR}")
    print(f"       목표엔트로피={TARGET_ENTROPY} | 초기α={np.exp(LOG_ALPHA_INIT):.3f}")

    env    = CartPoleEnv(XML_PATH)
    memory = ReplayMemory(MEMORY_SIZE)
    actor, critic1, critic2, critic1_target, critic2_target, log_alpha = init_networks()

    actor_opt  = torch.optim.Adam(actor.parameters(),   lr=ACTOR_LR)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=CRITIC_LR)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=CRITIC_LR)
    alpha_opt  = torch.optim.Adam([log_alpha],           lr=ALPHA_LR)

    best_reward = float("-inf")
    ep_rewards, ep_c_losses, ep_a_losses, ep_alphas = [], [], [], []

    sep = "=" * 75
    hdr = (f"{'에피소드':>10} | {'평균보상':>9} | {'Critic손실':>11} | "
           f"{'Actor손실':>10} | {'α':>7} | {'메모리':>7}")
    print(sep); print(hdr); print(sep)

    for ep in range(1, N_EPISODES + 1):
        state        = env.reset()
        total_reward = 0.0
        c_loss_sum   = 0.0
        a_loss_sum   = 0.0
        alpha_sum    = 0.0
        loss_cnt     = 0
        done         = False

        while not done:
            action                   = select_action(actor, state)
            next_state, reward, done = env.step(action)
            done_mask                = 0.0 if done else 1.0
            memory.put(state, action, reward, next_state, done_mask)
            state        = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                c_loss, a_loss, alpha_val = train_step(
                    actor, critic1, critic2, critic1_target, critic2_target,
                    memory, actor_opt, critic1_opt, critic2_opt, log_alpha, alpha_opt,
                )
                c_loss_sum += c_loss
                a_loss_sum += a_loss
                alpha_sum  += alpha_val
                loss_cnt   += 1

        avg_c = c_loss_sum / loss_cnt if loss_cnt > 0 else float("nan")
        avg_a = a_loss_sum / loss_cnt if loss_cnt > 0 else float("nan")
        avg_alpha = alpha_sum / loss_cnt if loss_cnt > 0 else np.exp(LOG_ALPHA_INIT)

        ep_rewards.append(total_reward)
        ep_c_losses.append(avg_c   if loss_cnt > 0 else 0.0)
        ep_a_losses.append(avg_a   if loss_cnt > 0 else 0.0)
        ep_alphas.append(avg_alpha)

        # 보상이 개선된 경우 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            actor.save_weights(MODEL_ACTOR_PATH)
            critic1.save_weights(MODEL_CRITIC1_PATH)
            critic2.save_weights(MODEL_CRITIC2_PATH)
            print(f"  ↑ [모델 저장] ep={ep:4d}  보상={total_reward:.2f}  α={avg_alpha:.4f}")

        if ep % PRINT_INTERVAL == 0:
            recent_r  = ep_rewards[-PRINT_INTERVAL:]
            recent_cl = [l for l in ep_c_losses[-PRINT_INTERVAL:] if l > 0]
            recent_al = [l for l in ep_a_losses[-PRINT_INTERVAL:] if l != 0.0]
            recent_alpha = [a for a in ep_alphas[-PRINT_INTERVAL:]]
            avg_r  = np.mean(recent_r)
            avg_cl = np.mean(recent_cl)    if recent_cl    else float("nan")
            avg_al = np.mean(recent_al)    if recent_al    else float("nan")
            avg_al_val = np.mean(recent_alpha)
            print(f"{ep:>10d} | {avg_r:>9.2f} | {avg_cl:>11.6f} | "
                  f"{avg_al:>10.6f} | {avg_al_val:>7.4f} | {len(memory):>7d}")

    print(sep)
    print("[학습 완료]")
    save_plots(ep_rewards, ep_c_losses, ep_a_losses, ep_alphas)


if __name__ == "__main__":
    main()
