"""
torch 기반 3파전 비교:
  1) true_sgd   : 진짜 gradient(autograd) 그대로 SGD
  2) zo_sgd     : 매 라운드 랜덤 방향 z(k개, shared 아님 — 라운드마다 새로 샘플)로
                  진짜 gradient를 projection한 pseudo_grad = (1/k) * sum_i (g·z_i) z_i
                  를 만들어서 momentum 없이 SGD step
  3) zo_adam    : 같은 pseudo_grad를 Adam(beta1/beta2/eps)으로 step

핵심 설계 포인트 (mechuri의 proj_adam 아이디어 그대로):
  - "forward-backward 한 번으로 진짜 gradient g를 구하고, 그 g를 k개 랜덤 방향에
    project해서 (seed, scalar) k개만 통신"하는 구조.
  - finite-difference(SPSA)가 아니라 g와 z의 정확한 내적이므로 mu-bias/curvature
    문제가 없음 — 순수하게 "몇 개 방향으로 압축했을 때 정보가 얼마나 남는가"만 남음.
  - pseudo_grad의 스케일을 1/k로 정규화해 둠 (k↑일수록 true gradient에 수렴하는
    unbiased estimator가 되도록. E[pseudo_grad] = g, k→∞에서 Var→0)

objective는 기본으로 2D rotated anisotropic quadratic (조건수 조절 가능)를 쓰지만,
loss_fn(x)->scalar 형태만 지키면 임의 차원/함수로 바로 교체 가능 (아래 make_objective 참고).

사용 예:
  python zo_direction_compare.py --k 1 --beta1 0.9 --n_steps 300
  python zo_direction_compare.py --k 8 --beta1 0.99 --dim 2 --cond 50
  python zo_direction_compare.py --k 32 --dim 100 --cond 20 --n_steps 2000   # 고차원으로 확장
  python zo_direction_compare.py --k 4 --beta1 0.9 --save_mp4 out.mp4        # 2D일 때만 mp4 가능
"""

import argparse
import math

import torch


# ----------------------------------------------------------------------------
# objective: rotated anisotropic quadratic, 임의 차원 d로 일반화
#   f(x) = 0.5 * x^T H x,  H = Q diag(eigs) Q^T
#   eigs를 [1, cond]에 log-spaced로 뿌려서 조건수(cond)를 직접 제어
# ----------------------------------------------------------------------------
def make_objective(dim: int, cond: float, seed: int = 0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # 랜덤 직교행렬 Q: QR decomposition of a random matrix
    A = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(A)
    eigs = torch.logspace(0, math.log10(cond), dim)  # [1, cond]
    H = (Q * eigs.unsqueeze(0)) @ Q.T
    H = H.to(device)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ H @ x

    return loss_fn, H


# ----------------------------------------------------------------------------
# pseudo-gradient: k개 랜덤 방향에 "진짜 gradient"를 projection
# ----------------------------------------------------------------------------
def true_grad(loss_fn, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    loss = loss_fn(x)
    (g,) = torch.autograd.grad(loss, x)
    return g.detach(), loss.detach()


def zo_pseudo_grad(g_true: torch.Tensor, k: int, device, dtype) -> torch.Tensor:
    """d차원 g_true를 k개의 iid N(0,I_d) 방향에 project해서 재구성.
    E[pseudo_grad] = g_true (k에 무관, unbiased), Var ∝ (d-1)/k * ||g_true||^2 / d 정도로
    k가 작을수록/차원이 클수록 노이즈가 커짐 (대화에서 다룬 cos ≈ sqrt(k/d) 관계)."""
    d = g_true.numel()
    Z = torch.randn(k, d, device=device, dtype=dtype)  # (k, d), 매 라운드 새로 샘플
    scalars = Z @ g_true  # (k,)  = g^T z_i, 각각 정확한 내적 (노이즈 없음)
    pseudo = (scalars.unsqueeze(1) * Z).sum(dim=0) / k  # (1/k) sum_i (g.z_i) z_i
    return pseudo


# ----------------------------------------------------------------------------
# 세 optimizer 루프
# ----------------------------------------------------------------------------
def run_true_sgd(loss_fn, x0, n_steps, lr):
    x = x0.clone()
    traj = [x.clone()]
    losses = [loss_fn(x).item()]
    for _ in range(n_steps):
        g, _ = true_grad(loss_fn, x)
        x = x - lr * g
        traj.append(x.clone())
        losses.append(loss_fn(x).item())
    return torch.stack(traj), losses


def run_zo_sgd(loss_fn, x0, n_steps, lr, k, seed=0):
    device, dtype = x0.device, x0.dtype
    torch.manual_seed(seed)
    x = x0.clone()
    traj = [x.clone()]
    losses = [loss_fn(x).item()]
    cos_hist = []
    for _ in range(n_steps):
        g_true, _ = true_grad(loss_fn, x)
        pseudo = zo_pseudo_grad(g_true, k, device, dtype)
        cos_hist.append(torch.nn.functional.cosine_similarity(
            pseudo.unsqueeze(0), g_true.unsqueeze(0)).item())
        x = x - lr * pseudo
        traj.append(x.clone())
        losses.append(loss_fn(x).item())
    return torch.stack(traj), losses, cos_hist


def run_zo_adam(loss_fn, x0, n_steps, lr, k, beta1, beta2, eps=1e-8, seed=0):
    device, dtype = x0.device, x0.dtype
    torch.manual_seed(seed)
    x = x0.clone()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    traj = [x.clone()]
    losses = [loss_fn(x).item()]
    cos_hist = []
    for t in range(1, n_steps + 1):
        g_true, _ = true_grad(loss_fn, x)
        pseudo = zo_pseudo_grad(g_true, k, device, dtype)
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
# 선택: 2D일 때만 mp4로 궤적 시각화 (등고선 + 세 궤적 + cosine curve)
# ----------------------------------------------------------------------------
def save_trajectory_mp4(H, traj_true, traj_zo_sgd, traj_zo_adam,
                        cos_zo_sgd, cos_zo_adam, out_path, fps=20):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    H_np = H.cpu().numpy()
    trajs = {
        "true_sgd": traj_true.cpu().numpy(),
        "zo_sgd": traj_zo_sgd.cpu().numpy(),
        "zo_adam": traj_zo_adam.cpu().numpy(),
    }
    coses = {
        "true_sgd": np.ones(len(traj_true) - 1),
        "zo_sgd": np.array(cos_zo_sgd),
        "zo_adam": np.array(cos_zo_adam),
    }
    colors = {"true_sgd": "#58a6ff", "zo_sgd": "#f85149", "zo_adam": "#3fb950"}
    labels = {"true_sgd": "True SGD", "zo_sgd": "ZO SGD (no momentum)", "zo_adam": "ZO + Adam"}

    n_steps = max(len(t) for t in trajs.values()) - 1
    lim = max(np.abs(np.concatenate([t.flatten() for t in trajs.values()]))) * 1.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.7),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")

    gx = np.linspace(-lim, lim, 300)
    gy = np.linspace(-lim, lim, 300)
    GX, GY = np.meshgrid(gx, gy)
    GZ = 0.5 * (H_np[0, 0] * GX**2 + 2 * H_np[0, 1] * GX * GY + H_np[1, 1] * GY**2)
    ax1.contour(GX, GY, GZ, levels=np.geomspace(max(GZ.max() * 1e-4, 1e-6), GZ.max(), 18),
               cmap="cool", linewidths=0.6, alpha=0.55)
    ax1.plot(0, 0, marker="*", color="white", markersize=14, zorder=5)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_title("trajectory", color="white")
    ax1.tick_params(colors="#8b949e")

    lines, points = {}, {}
    for key in trajs:
        (line,) = ax1.plot([], [], color=colors[key], lw=1.6, alpha=0.85, label=labels[key])
        (pt,) = ax1.plot([], [], marker="o", color=colors[key], markersize=7, zorder=6)
        lines[key], points[key] = line, pt
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)

    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.set_title("cos(pseudo_grad, true grad)", color="white")
    ax2.set_xlabel("round", color="#8b949e")
    ax2.tick_params(colors="#8b949e")

    cos_lines = {}
    for key in trajs:
        (cline,) = ax2.plot([], [], color=colors[key], lw=1.8)
        cos_lines[key] = cline

    def init():
        for key in trajs:
            lines[key].set_data([], [])
            points[key].set_data([], [])
            cos_lines[key].set_data([], [])
        return list(lines.values()) + list(points.values()) + list(cos_lines.values())

    def update(frame):
        for key in trajs:
            t = trajs[key]
            idx = min(frame, len(t) - 1)
            lines[key].set_data(t[: idx + 1, 0], t[: idx + 1, 1])
            points[key].set_data([t[idx, 0]], [t[idx, 1]])
            c = coses[key]
            cidx = min(frame, len(c))
            cos_lines[key].set_data(np.arange(cidx), c[:cidx])
        return list(lines.values()) + list(points.values()) + list(cos_lines.values())

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_steps + 1, interval=1000 / fps, blit=True)
    plt.tight_layout()
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=140)
    print(f"[viz] saved -> {out_path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2, help="목적함수 차원. mp4 시각화는 dim=2에서만 지원")
    p.add_argument("--cond", type=float, default=50.0, help="Hessian 조건수 (조절 가능한 ill-conditioning)")
    p.add_argument("--k", type=int, default=1, help="랜덤 방향 개수 (통신량에 해당)")
    p.add_argument("--n_steps", type=int, default=300)
    p.add_argument("--lr_true", type=float, default=0.018)
    p.add_argument("--lr_zo_sgd", type=float, default=0.10)
    p.add_argument("--lr_zo_adam", type=float, default=0.055)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=1, help="ZO 방향 샘플링 seed (zo_sgd/zo_adam 공통)")
    p.add_argument("--obj_seed", type=int, default=0, help="목적함수(H) 생성 seed")
    p.add_argument("--start_scale", type=float, default=2.0, help="시작점 x0의 스케일")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--save_mp4", default=None, help="dim=2일 때만: mp4 저장 경로")
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float32

    loss_fn, H = make_objective(args.dim, args.cond, seed=args.obj_seed, device=device)

    g0 = torch.Generator(device="cpu").manual_seed(42)
    x0 = torch.randn(args.dim, generator=g0).to(device=device, dtype=dtype)
    x0 = x0 / x0.norm() * args.start_scale

    traj_true, loss_true = run_true_sgd(loss_fn, x0, args.n_steps, args.lr_true)
    traj_zs, loss_zs, cos_zs = run_zo_sgd(loss_fn, x0, args.n_steps, args.lr_zo_sgd,
                                          args.k, seed=args.seed)
    traj_za, loss_za, cos_za = run_zo_adam(loss_fn, x0, args.n_steps, args.lr_zo_adam,
                                           args.k, args.beta1, args.beta2, seed=args.seed)

    print(f"[dim={args.dim} cond={args.cond:g} k={args.k} steps={args.n_steps}]")
    print(f"  true_sgd : final loss = {loss_true[-1]:.6e}")
    print(f"  zo_sgd   : final loss = {loss_zs[-1]:.6e}   mean cos = {sum(cos_zs)/len(cos_zs):.4f}")
    print(f"  zo_adam  : final loss = {loss_za[-1]:.6e}   mean cos = {sum(cos_za)/len(cos_za):.4f}")

    if args.save_mp4:
        if args.dim != 2:
            print("[warn] mp4 시각화는 dim=2에서만 지원 (H를 2x2로 그림) — 저장 건너뜀")
        else:
            save_trajectory_mp4(H, traj_true, traj_zs, traj_za, cos_zs, cos_za, args.save_mp4)
