"""
SeedFlood(ZO) vs FO SGD convergence test - ResNet-18, CIFAR-10, iid partition, n_nodes simulated sequentially.
Run: python seedflood_convergence_test.py
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import json
from collections import deque
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---------------- data ----------------
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=512, shuffle=False)

def partition_iid(dataset, n_nodes, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))
    return [idx[i::n_nodes] for i in range(n_nodes)]

N_NODES = 4
BATCH_SIZE = 32
node_indices = partition_iid(trainset, N_NODES)
node_loaders = [DataLoader(Subset(trainset, idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                for idx in node_indices]

# ---------------- model ----------------
def make_model():
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    # cifar-sized stem (32x32 input, no aggressive downsample)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m.to(device)

loss_fn_ce = nn.CrossEntropyLoss()

def eval_test_loss(model, n_batches=10):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= n_batches: break
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += loss_fn_ce(out, y).item()
            count += 1
    model.train()
    return total / count

# ---------------- ZO (SeedFlood-style SPSA) ----------------
@torch.no_grad()
def zo_step(model, batch, mu=1e-3, seed=None):
    x, y = batch
    x, y = x.to(device), y.to(device)
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    params = [p for p in model.parameters() if p.requires_grad]

    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=mu)
    loss_pos = loss_fn_ce(model(x), y).item()

    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=-2 * mu)
    loss_neg = loss_fn_ce(model(x), y).item()

    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=mu)  # restore

    scalar = (loss_pos - loss_neg) / (2 * mu)
    return seed, scalar, (loss_pos + loss_neg) / 2

@torch.no_grad()
def apply_zo_update(model, node_results, lr, mode="sign"):
    # mode="sign": scalar 크기 버리고 방향(+z or -z)만 사용 -> sign-ZO / ZO-signSGD
    # mode="norm": scalar를 이번 round node들 magnitude로 나눠서 정규화 (상대적 방향 강도는 유지, outlier만 억제)
    # mode="raw": 원래 SPSA scalar 그대로 사용
    if mode == "norm":
        mags = [abs(s) for _, s in node_results]
        denom = max(mags) if max(mags) > 0 else 1.0
        steps = [(seed, s / denom) for seed, s in node_results]
    elif mode == "sign":
        steps = [(seed, np.sign(s)) for seed, s in node_results]
    else:
        steps = node_results

    for seed, step in steps:
        if step == 0:
            continue
        torch.manual_seed(seed)
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn_like(p)
                p.add_(z, alpha=-lr * step / len(steps))

@torch.no_grad()
def reconstruct_pseudo_grad(model, node_results, mode="raw"):
    """server-side: seed로 z 재생성해서 평균 pseudo-gradient dict 구성. 통신량엔 영향 없음 (server 내부 연산)."""
    if mode == "sign":
        steps = [(seed, np.sign(s)) for seed, s in node_results]
    elif mode == "norm":
        mags = [abs(s) for _, s in node_results]
        denom = max(mags) if max(mags) > 0 else 1.0
        steps = [(seed, s / denom) for seed, s in node_results]
    else:
        steps = node_results

    pseudo_grad = {id(p): torch.zeros_like(p) for p in model.parameters() if p.requires_grad}
    for seed, step in steps:
        if step == 0:
            continue
        torch.manual_seed(seed)
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn_like(p)
                pseudo_grad[id(p)].add_(z, alpha=step / len(steps))
    return pseudo_grad

@torch.no_grad()
def adam_zo_update(model, pseudo_grad, adam_state, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    beta1, beta2 = betas
    for p in model.parameters():
        if not p.requires_grad:
            continue
        g = pseudo_grad[id(p)]
        st = adam_state[id(p)]
        st["m"].mul_(beta1).add_(g, alpha=1 - beta1)
        st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
        m_hat = st["m"] / (1 - beta1 ** t)
        v_hat = st["v"] / (1 - beta2 ** t)
        p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

def train_zo_adam(n_rounds=3000, lr=1e-3, mu=1e-3, betas=(0.9, 0.999), log_every=100,
                   grad_recon_mode="raw"):
    """ZO scalar 그대로(raw) 재구성해서 server-side Adam moment로 스케일/노이즈 흡수.
    grad_recon_mode: 'raw'(scalar 그대로) / 'sign' / 'norm' 중 pseudo-grad 재구성 방식 선택."""
    model = make_model()
    adam_state = {id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                  for p in model.parameters() if p.requires_grad}
    node_iters = [iter(dl) for dl in node_loaders]
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(1, n_rounds + 1), desc="ZO-Adam")
    for r in pbar:
        node_results = []
        approx_losses = []
        for i, dl in enumerate(node_loaders):
            try:
                batch = next(node_iters[i])
            except StopIteration:
                node_iters[i] = iter(dl)
                batch = next(node_iters[i])
            seed, scalar, approx_loss = zo_step(model, batch, mu=mu)
            node_results.append((seed, scalar))
            approx_losses.append(approx_loss)

        pseudo_grad = reconstruct_pseudo_grad(model, node_results, mode=grad_recon_mode)
        adam_zo_update(model, pseudo_grad, adam_state, t=r, lr=lr, betas=betas)

        roll.append(float(np.mean(approx_losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds:
            test_loss = eval_test_loss(model)
            elapsed = time.time() - t0
            log.append({"round": r, "train_approx": float(np.mean(approx_losses)),
                        "train_avg100": float(np.mean(roll)), "test_loss": test_loss, "time": elapsed})
    return log

def train_zo(n_rounds=3000, lr=1e-2, mu=1e-3, log_every=100, update_mode="sign"):
    model = make_model()
    node_iters = [iter(dl) for dl in node_loaders]
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(n_rounds), desc="ZO")
    for r in pbar:
        node_results = []
        approx_losses = []
        for i, dl in enumerate(node_loaders):
            try:
                batch = next(node_iters[i])
            except StopIteration:
                node_iters[i] = iter(dl)
                batch = next(node_iters[i])
            seed, scalar, approx_loss = zo_step(model, batch, mu=mu)
            node_results.append((seed, scalar))
            approx_losses.append(approx_loss)
        apply_zo_update(model, node_results, lr, mode=update_mode)

        roll.append(float(np.mean(approx_losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds - 1:
            test_loss = eval_test_loss(model)
            elapsed = time.time() - t0
            log.append({"round": r, "train_approx": float(np.mean(approx_losses)),
                        "train_avg100": float(np.mean(roll)), "test_loss": test_loss, "time": elapsed})
    return log

# ---------------- FO SGD baseline ----------------
def train_fo(n_rounds=3000, lr=1e-2, log_every=100):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    node_iters = [iter(dl) for dl in node_loaders]
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(n_rounds), desc="FO")
    for r in pbar:
        # average grads across nodes (like DDP all-reduce), single param update
        opt.zero_grad()
        losses = []
        for i, dl in enumerate(node_loaders):
            try:
                x, y = next(node_iters[i])
            except StopIteration:
                node_iters[i] = iter(dl)
                x, y = next(node_iters[i])
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn_ce(out, y) / N_NODES
            loss.backward()
            losses.append(loss.item() * N_NODES)
        opt.step()

        roll.append(float(np.mean(losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds - 1:
            test_loss = eval_test_loss(model)
            elapsed = time.time() - t0
            log.append({"round": r, "train_loss": float(np.mean(losses)),
                        "train_avg100": float(np.mean(roll)), "test_loss": test_loss, "time": elapsed})
    return log

if __name__ == "__main__":
    N_ROUNDS = 3000  # ZO 논문 기준 FO의 ~5-10x iter 필요하다고 봄

    print("=== FO SGD baseline ===")
    fo_log = train_fo(n_rounds=N_ROUNDS, lr=0.01)

    print("\n=== SeedFlood ZO (sign) ===")
    zo_log = train_zo(n_rounds=N_ROUNDS * 5, lr=0.01, mu=1e-3, update_mode="sign")

    print("\n=== SeedFlood ZO-Adam (raw scalar + server-side moments) ===")
    zo_adam_log = train_zo_adam(n_rounds=N_ROUNDS * 5, lr=1e-3, mu=1e-3, grad_recon_mode="raw")

    with open("convergence_results.json", "w") as f:
        json.dump({"fo": fo_log, "zo": zo_log, "zo_adam": zo_adam_log}, f, indent=2)
    print("\nsaved to convergence_results.json")
