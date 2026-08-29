"""
zo_direction_compare.py의 3파전(true_sgd / zo_sgd / zo_adam)을 dim=200에서 돌리고,
true_sgd 궤적을 기준으로 PCA top-2 평면을 뽑아서 세 궤적을 그 위에 투영해 애니메이션.

PCA basis를 true_sgd에서 뽑는 이유: true gradient가 실제로 밟는 경로가 landscape의
"진짜 정보가 있는" 방향(주로 sharp/steep 고유벡터 쪽으로 빠르게, 이후 완만한 방향으로
서서히)이므로, 이 경로의 분산이 가장 큰 2개 방향이 전체 최적화의 핵심 축일 가능성이 높음.
zo_sgd/zo_adam은 같은 basis에 투영해서 "같은 평면에서 true GD와 얼마나 다르게 움직이는가"를
직접 비교 가능하게 함 (각자 다른 PCA를 쓰면 축 자체가 달라져서 비교가 안 됨).
"""

import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------------------------------------------------------
# 목적함수: rotated anisotropic quadratic, dim=200
# ----------------------------------------------------------------------------
DIM = 200
COND = 40.0
N_STEPS = 1500
K = 8
BETA1, BETA2, EPS = 0.99, 0.999, 1e-8
LR_TRUE = 0.02
LR_ZO_SGD = 0.06
LR_ZO_ADAM = 0.02
DEVICE = "cpu"
DTYPE = torch.float32


def make_objective(dim, cond, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(A)
    eigs = torch.logspace(0, math.log10(cond), dim)
    H = (Q * eigs.unsqueeze(0)) @ Q.T
    return H


H = make_objective(DIM, COND, seed=0).to(DEVICE)


def loss_fn(x):
    return 0.5 * x @ H @ x


def true_grad(x):
    x = x.detach().requires_grad_(True)
    loss = loss_fn(x)
    (g,) = torch.autograd.grad(loss, x)
    return g.detach(), loss.detach()


def zo_pseudo_grad(g_true, k):
    d = g_true.numel()
    Z = torch.randn(k, d, device=DEVICE, dtype=DTYPE)
    scalars = Z @ g_true
    return (scalars.unsqueeze(1) * Z).sum(dim=0) / k


def run_true_sgd(x0, n_steps, lr):
    x = x0.clone()
    traj, losses = [x.clone()], [loss_fn(x).item()]
    for _ in range(n_steps):
        g, _ = true_grad(x)
        x = x - lr * g
        traj.append(x.clone())
        losses.append(loss_fn(x).item())
    return torch.stack(traj), losses


def run_zo_sgd(x0, n_steps, lr, k, seed=1):
    torch.manual_seed(seed)
    x = x0.clone()
    traj, losses, cos_hist = [x.clone()], [loss_fn(x).item()], []
    for _ in range(n_steps):
        g_true, _ = true_grad(x)
        pseudo = zo_pseudo_grad(g_true, k)
        cos_hist.append(torch.nn.functional.cosine_similarity(
            pseudo.unsqueeze(0), g_true.unsqueeze(0)).item())
        x = x - lr * pseudo
        traj.append(x.clone())
        losses.append(loss_fn(x).item())
    return torch.stack(traj), losses, cos_hist


def run_zo_adam(x0, n_steps, lr, k, beta1, beta2, eps=1e-8, seed=1):
    torch.manual_seed(seed)
    x = x0.clone()
    m, v = torch.zeros_like(x), torch.zeros_like(x)
    traj, losses, cos_hist = [x.clone()], [loss_fn(x).item()], []
    for t in range(1, n_steps + 1):
        g_true, _ = true_grad(x)
        pseudo = zo_pseudo_grad(g_true, k)
        cos_hist.append(torch.nn.functional.cosine_similarity(
            pseudo.unsqueeze(0), g_true.unsqueeze(0)).item())
        m = beta1 * m + (1 - beta1) * pseudo
        v = beta2 * v + (1 - beta2) * pseudo.pow(2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (v_hat.sqrt() + eps)
        x = x - lr * update
        traj.append(x.clone())
        losses.append(loss_fn(x).item())
    return torch.stack(traj), losses, cos_hist


# ----------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------
g0 = torch.Generator(device="cpu").manual_seed(42)
x0 = torch.randn(DIM, generator=g0).to(DEVICE, DTYPE)
x0 = x0 / x0.norm() * 3.0

print("[run] true_sgd ...")
traj_true, loss_true = run_true_sgd(x0, N_STEPS, LR_TRUE)
print("[run] zo_sgd ...")
traj_zs, loss_zs, cos_zs = run_zo_sgd(x0, N_STEPS, LR_ZO_SGD, K, seed=1)
print("[run] zo_adam ...")
traj_za, loss_za, cos_za = run_zo_adam(x0, N_STEPS, LR_ZO_ADAM, K, BETA1, BETA2, seed=1)

print(f"[dim={DIM} cond={COND:g} k={K} steps={N_STEPS}]")
print(f"  true_sgd : final loss = {loss_true[-1]:.6e}")
print(f"  zo_sgd   : final loss = {loss_zs[-1]:.6e}   mean cos = {sum(cos_zs)/len(cos_zs):.4f}")
print(f"  zo_adam  : final loss = {loss_za[-1]:.6e}   mean cos = {sum(cos_za)/len(cos_za):.4f}")

# ----------------------------------------------------------------------------
# PCA: true_sgd 궤적 기준으로 top-2 축 추출, 세 궤적 모두 투영
# ----------------------------------------------------------------------------
T = traj_true.numpy()  # (N+1, DIM)
mean = T.mean(axis=0)
Tc = T - mean
U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
pc = Vt[:2]  # (2, DIM) top-2 principal directions
explained = (S[:2] ** 2).sum() / (S ** 2).sum()
print(f"[pca] top-2 explained variance ratio (of true_sgd traj) = {explained:.4f}")


def project(traj_np):
    return (traj_np - mean) @ pc.T  # (N+1, 2)


proj_true = project(traj_true.numpy())
proj_zs = project(traj_zs.numpy())
proj_za = project(traj_za.numpy())

# ----------------------------------------------------------------------------
# 3-panel 애니메이션: PCA plane trajectory / loss curve(log) / cosine sim
# ----------------------------------------------------------------------------
colors = {"true": "#58a6ff", "zs": "#f85149", "za": "#3fb950"}
labels = {"true": "True SGD", "zs": "ZO SGD (no momentum)", "za": "ZO + Adam"}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.4),
                                    gridspec_kw={"width_ratios": [1.2, 1, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_color("#30363d")

# panel 1: PCA plane
all_pts = np.concatenate([proj_true, proj_zs, proj_za], axis=0)
lim = np.abs(all_pts).max() * 1.1
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_aspect("equal")
ax1.set_title(f"PCA plane of true_sgd traj (top-2, {explained*100:.1f}% var)", color="white", fontsize=10.5)
ax1.set_xlabel("PC1", color="#8b949e")
ax1.set_ylabel("PC2", color="#8b949e")
ax1.plot(proj_true[0, 0], proj_true[0, 1], marker="x", color="white", markersize=10, zorder=5)

lines1, points1 = {}, {}
for key, proj in (("true", proj_true), ("zs", proj_zs), ("za", proj_za)):
    (line,) = ax1.plot([], [], color=colors[key], lw=1.4, alpha=0.85, label=labels[key])
    (pt,) = ax1.plot([], [], marker="o", color=colors[key], markersize=6, zorder=6)
    lines1[key], points1[key] = line, pt
ax1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
          labelcolor="white", fontsize=8.5)

# panel 2: loss curve (log scale, y축 clip해서 0 근처도 보이게)
ax2.set_xlim(0, N_STEPS)
losses_all = np.array(loss_true + loss_zs + loss_za)
ymin = max(losses_all[losses_all > 0].min() * 0.5, 1e-8)
ymax = losses_all.max() * 1.3
ax2.set_ylim(ymin, ymax)
ax2.set_yscale("log")
ax2.set_title("loss (log scale)", color="white", fontsize=10.5)
ax2.set_xlabel("round", color="#8b949e")
lines2 = {}
for key, ls in (("true", loss_true), ("zs", loss_zs), ("za", loss_za)):
    (line,) = ax2.plot([], [], color=colors[key], lw=1.8)
    lines2[key] = line

# panel 3: cosine similarity
ax3.set_xlim(0, N_STEPS)
ax3.set_ylim(-1.05, 1.05)
ax3.axhline(0, color="#30363d", lw=0.8)
ax3.set_title("cos(pseudo_grad, true grad)", color="white", fontsize=10.5)
ax3.set_xlabel("round", color="#8b949e")
lines3 = {}
for key, cs in (("zs", cos_zs), ("za", cos_za)):
    (line,) = ax3.plot([], [], color=colors[key], lw=1.8)
    lines3[key] = line

round_text = ax1.text(0.02, 0.02, "", transform=ax1.transAxes, color="white",
                      fontsize=9, family="monospace")

data_traj = {"true": proj_true, "zs": proj_zs, "za": proj_za}
data_loss = {"true": loss_true, "zs": loss_zs, "za": loss_za}
data_cos = {"zs": cos_zs, "za": cos_za}

N_FRAMES = 300  # 1500 step을 300 frame으로 서브샘플 (mp4 길이/용량 조절)
frame_rounds = np.linspace(0, N_STEPS, N_FRAMES, dtype=int)


def init():
    artists = []
    for key in ("true", "zs", "za"):
        lines1[key].set_data([], [])
        points1[key].set_data([], [])
        lines2[key].set_data([], [])
        artists += [lines1[key], points1[key], lines2[key]]
    for key in ("zs", "za"):
        lines3[key].set_data([], [])
        artists.append(lines3[key])
    round_text.set_text("")
    artists.append(round_text)
    return artists


def update(frame_idx):
    r = frame_rounds[frame_idx]
    artists = []
    for key in ("true", "zs", "za"):
        p = data_traj[key]
        idx = min(r, len(p) - 1)
        lines1[key].set_data(p[: idx + 1, 0], p[: idx + 1, 1])
        points1[key].set_data([p[idx, 0]], [p[idx, 1]])
        l = data_loss[key]
        lidx = min(r, len(l) - 1)
        lines2[key].set_data(np.arange(lidx + 1), l[: lidx + 1])
        artists += [lines1[key], points1[key], lines2[key]]
    for key in ("zs", "za"):
        c = data_cos[key]
        cidx = min(r, len(c))
        lines3[key].set_data(np.arange(cidx), c[:cidx])
        artists.append(lines3[key])
    round_text.set_text(f"round {r:5d}")
    artists.append(round_text)
    return artists


anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=N_FRAMES, interval=1000 / 20, blit=True)
plt.tight_layout()

out_path = "./zo_pca_200d.mp4"
anim.save(out_path, writer=animation.FFMpegWriter(fps=20, bitrate=2600), dpi=140)
print(f"[viz] saved -> {out_path}")
