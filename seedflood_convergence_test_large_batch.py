"""
SeedFlood(ZO) vs FO SGD convergence test - ResNet-18, CIFAR-100 (HF uoft-cs/cifar100),
iid partition, n_nodes simulated sequentially.
Run: python seedflood_cifar100.py
"""
import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
import json
from collections import deque
from tqdm import tqdm
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---------------- data (전체를 GPU 상주 uint8 텐서로, DataLoader 제거) ----------------
MEAN = torch.tensor([0.5071, 0.4865, 0.4409], device=device).view(1, 3, 1, 1)
STD  = torch.tensor([0.2673, 0.2564, 0.2762], device=device).view(1, 3, 1, 1)

def load_split(split):
    imgs = np.stack([np.asarray(im) for im in split["img"]])          # N,32,32,3 uint8
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device)  # N,3,32,32 uint8 (~150MB)
    y = torch.tensor(split["fine_label"], dtype=torch.long, device=device)
    return x, y

print("loading uoft-cs/cifar100 ...")
ds = load_dataset("uoft-cs/cifar100")
train_x, train_y = load_split(ds["train"])
test_x, test_y = load_split(ds["test"])
print(f"train {tuple(train_x.shape)}, test {tuple(test_x.shape)}")

def normalize(x_uint8):
    return (x_uint8.float().div_(255.0) - MEAN) / STD

def partition_iid(n_total, n_nodes, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    return [idx[i::n_nodes] for i in range(n_nodes)]

class NodeSampler:
    """DataLoader 대체: GPU 텐서 인덱싱으로 배치 생성 (worker/collate 오버헤드 제거)."""
    def __init__(self, indices, batch_size, seed):
        self.idx = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=device)
        self.bs = batch_size
        self.n = len(indices)
        self.g = torch.Generator().manual_seed(seed)  # cpu generator (randperm용)
        self._new_epoch()

    def _new_epoch(self):
        perm = torch.randperm(self.n, generator=self.g).to(device)
        self.perm = self.idx[perm]
        self.pos = 0

    def next_batch(self):
        if self.pos + self.bs > self.n:  # drop_last와 동일
            self._new_epoch()
        sel = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return normalize(train_x[sel]), train_y[sel]

N_NODES = 4
BATCH_SIZE = 128
node_samplers_seedbase = 1000

def make_node_samplers():
    parts = partition_iid(len(train_x), N_NODES)
    return [NodeSampler(p, BATCH_SIZE, seed=node_samplers_seedbase + i) for i, p in enumerate(parts)]

# ---------------- model ----------------
def make_model(norm="gn"):
    # ZO는 매 step model 파라미터를 ±mu로 흔든 상태에서 두 번 forward하는데,
    # BatchNorm(train mode)은 그 두 forward마다 running stats를 오염시킴 → GroupNorm이 안전 (federated 표준이기도 함)
    kwargs = {}
    if norm == "gn":
        kwargs["norm_layer"] = lambda ch: nn.GroupNorm(32, ch)
    m = torchvision.models.resnet18(weights=None, num_classes=100, **kwargs)
    # cifar-sized stem (32x32 input, no aggressive downsample)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m.to(device)

loss_fn_ce = nn.CrossEntropyLoss()

@torch.no_grad()
def eval_test(model, batch_size=1000):
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(test_x), batch_size):
        x = normalize(test_x[i:i + batch_size])
        y = test_y[i:i + batch_size]
        out = model(x)
        total_loss += loss_fn_ce(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
    model.train()
    return total_loss / len(test_x), correct / len(test_x)

# ---------------- ZO (SeedFlood-style SPSA) ----------------
def _regen(params, seed):
    """seed로 z 재생성. 전역 torch.manual_seed 대신 전용 Generator 사용
    (전역 RNG 오염 방지: seed chain이 deterministic해지는 문제 + 다른 랜덤 소스와 얽힘 제거)."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    for p in params:
        yield p, torch.randn(p.shape, dtype=p.dtype, device=device, generator=g)

@torch.no_grad()
def zo_step(model, batch, mu, seed):
    x, y = batch
    params = [p for p in model.parameters() if p.requires_grad]

    for p, z in _regen(params, seed):
        p.add_(z, alpha=mu)
    loss_pos = loss_fn_ce(model(x), y).item()

    for p, z in _regen(params, seed):
        p.add_(z, alpha=-2 * mu)
    loss_neg = loss_fn_ce(model(x), y).item()

    for p, z in _regen(params, seed):
        p.add_(z, alpha=mu)  # restore

    scalar = (loss_pos - loss_neg) / (2 * mu)
    return scalar, (loss_pos + loss_neg) / 2

def _resolve_steps(node_results, mode):
    if mode == "norm":
        mags = [abs(s) for _, s in node_results]
        denom = max(mags) if max(mags) > 0 else 1.0
        return [(seed, s / denom) for seed, s in node_results]
    elif mode == "sign":
        return [(seed, float(np.sign(s))) for seed, s in node_results]
    return node_results

@torch.no_grad()
def apply_zo_update(model, node_results, lr, mode="sign"):
    # mode="sign": scalar 크기 버리고 방향(+z or -z)만 사용 -> sign-ZO / ZO-signSGD
    # mode="norm": scalar를 이번 round node들 magnitude로 나눠서 정규화
    # mode="raw": 원래 SPSA scalar 그대로 사용
    steps = _resolve_steps(node_results, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    for seed, step in steps:
        if step == 0:
            continue
        for p, z in _regen(params, seed):
            p.add_(z, alpha=-lr * step / len(steps))

@torch.no_grad()
def reconstruct_pseudo_grad(params, node_results, buffers, mode="raw"):
    """server-side: seed로 z 재생성해서 평균 pseudo-gradient 구성.
    buffers는 round마다 재할당하지 않고 zero_() 후 재사용."""
    steps = _resolve_steps(node_results, mode)
    for buf in buffers:
        buf.zero_()
    for seed, step in steps:
        if step == 0:
            continue
        for (p, z), buf in zip(_regen(params, seed), buffers):
            buf.add_(z, alpha=step / len(steps))
    return buffers

@torch.no_grad()
def adam_zo_update(params, pseudo_grad, adam_m, adam_v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    beta1, beta2 = betas
    for p, g, m, v in zip(params, pseudo_grad, adam_m, adam_v):
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

def train_zo_adam(n_rounds=3000, lr=1e-3, mu=1e-3, betas=(0.9, 0.999), log_every=100,
                  grad_recon_mode="raw", run_seed=0):
    model = make_model()
    params = [p for p in model.parameters() if p.requires_grad]
    buffers = [torch.zeros_like(p) for p in params]
    adam_m = [torch.zeros_like(p) for p in params]
    adam_v = [torch.zeros_like(p) for p in params]
    samplers = make_node_samplers()
    seed_rng = np.random.default_rng(run_seed)
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(1, n_rounds + 1), desc="ZO-Adam")
    for r in pbar:
        node_results, approx_losses = [], []
        for sampler in samplers:
            seed = int(seed_rng.integers(0, 2**31 - 1))
            scalar, approx_loss = zo_step(model, sampler.next_batch(), mu=mu, seed=seed)
            node_results.append((seed, scalar))
            approx_losses.append(approx_loss)

        pseudo_grad = reconstruct_pseudo_grad(params, node_results, buffers, mode=grad_recon_mode)
        adam_zo_update(params, pseudo_grad, adam_m, adam_v, t=r, lr=lr, betas=betas)

        roll.append(float(np.mean(approx_losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds:
            test_loss, test_acc = eval_test(model)
            log.append({"round": r, "train_approx": float(np.mean(approx_losses)),
                        "train_avg100": float(np.mean(roll)),
                        "test_loss": test_loss, "test_acc": test_acc,
                        "time": time.time() - t0})
    return log

def train_zo(n_rounds=3000, lr=1e-2, mu=1e-3, log_every=100, update_mode="sign", run_seed=0):
    model = make_model()
    samplers = make_node_samplers()
    seed_rng = np.random.default_rng(run_seed)
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(n_rounds), desc="ZO")
    for r in pbar:
        node_results, approx_losses = [], []
        for sampler in samplers:
            seed = int(seed_rng.integers(0, 2**31 - 1))
            scalar, approx_loss = zo_step(model, sampler.next_batch(), mu=mu, seed=seed)
            node_results.append((seed, scalar))
            approx_losses.append(approx_loss)
        apply_zo_update(model, node_results, lr, mode=update_mode)

        roll.append(float(np.mean(approx_losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds - 1:
            test_loss, test_acc = eval_test(model)
            log.append({"round": r, "train_approx": float(np.mean(approx_losses)),
                        "train_avg100": float(np.mean(roll)),
                        "test_loss": test_loss, "test_acc": test_acc,
                        "time": time.time() - t0})
    return log

# ---------------- FO SGD baseline ----------------
def train_fo(n_rounds=3000, lr=1e-2, log_every=100):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    samplers = make_node_samplers()
    log = []
    roll = deque(maxlen=100)
    t0 = time.time()
    pbar = tqdm(range(n_rounds), desc="FO")
    for r in pbar:
        # average grads across nodes (like DDP all-reduce), single param update
        opt.zero_grad(set_to_none=True)
        losses = []
        for sampler in samplers:
            x, y = sampler.next_batch()
            loss = loss_fn_ce(model(x), y) / N_NODES
            loss.backward()
            losses.append(loss.item() * N_NODES)
        opt.step()

        roll.append(float(np.mean(losses)))
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")

        if r % log_every == 0 or r == n_rounds - 1:
            test_loss, test_acc = eval_test(model)
            log.append({"round": r, "train_loss": float(np.mean(losses)),
                        "train_avg100": float(np.mean(roll)),
                        "test_loss": test_loss, "test_acc": test_acc,
                        "time": time.time() - t0})
    return log

if __name__ == "__main__":
    N_ROUNDS = 3000  # ZO 논문 기준 FO의 ~5-10x iter 필요하다고 봄

    #print("=== FO SGD baseline ===")
    #fo_log = train_fo(n_rounds=N_ROUNDS, lr=0.01)

    #print("\n=== SeedFlood ZO (norm) ===")
    #zo_log = train_zo(n_rounds=N_ROUNDS * 5, lr=0.01, mu=1e-3, update_mode="norm")

    print("\n=== SeedFlood ZO-Adam (raw scalar + server-side moments) ===")
    zo_adam_log = train_zo_adam(n_rounds=N_ROUNDS * 5, lr=1e-3, mu=1e-3, grad_recon_mode="raw")

    with open("convergence_results.json", "w") as f:
        json.dump({"fo": fo_log, "zo": zo_log, "zo_adam": zo_adam_log}, f, indent=2)
    print("\nsaved to convergence_results_lb.json")
