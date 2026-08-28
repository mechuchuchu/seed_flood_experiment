"""
MNIST SeedFlood ZO testbed - fast lr/mu sweep.
Shared-seed-per-round variant:
  - server picks ONE seed per round, broadcasts
  - each node evaluates L(theta + mu*z), L(theta - mu*z) on its OWN batch (same z)
  - server averages L+ and L- across nodes -> single scalar = (Lbar+ - Lbar-) / (2mu)
  - pseudo_grad = scalar * z ; optionally through Adam

Run: python mnist_seedflood_sweep.py
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from collections import deque
from tqdm import tqdm
import itertools
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data ----------------
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

def partition_iid(dataset, n_nodes, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))
    return [idx[i::n_nodes] for i in range(n_nodes)]

N_NODES = 4
BATCH_SIZE = 64
node_indices = partition_iid(trainset, N_NODES)
node_loaders = [DataLoader(Subset(trainset, idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                for idx in node_indices]

# ---------------- tiny model (fast) ----------------
def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def eval_test_loss(model, n_batches=5):
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x), y).item()
        count += 1
    model.train()
    return total / count

@torch.no_grad()
def eval_test_acc(model, n_batches=5):
    model.eval()
    correct, total = 0, 0
    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total

# ---------------- shared-seed ZO round ----------------
@torch.no_grad()
def zo_round_shared_seed(model, node_batches, mu=1e-3, seed=None):
    """server picks 1 seed, all nodes perturb with SAME z, evaluate on own batch."""
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    params = [p for p in model.parameters() if p.requires_grad]

    # +mu*z
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=mu)
    losses_pos = []
    for x, y in node_batches:
        losses_pos.append(loss_fn(model(x), y).item())

    # -2mu*z (regenerate same z sequence)
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=-2 * mu)
    losses_neg = []
    for x, y in node_batches:
        losses_neg.append(loss_fn(model(x), y).item())

    # restore
    torch.manual_seed(seed)
    for p in params:
        z = torch.randn_like(p)
        p.add_(z, alpha=mu)

    Lbar_pos = float(np.mean(losses_pos))
    Lbar_neg = float(np.mean(losses_neg))
    scalar = (Lbar_pos - Lbar_neg) / (2 * mu)
    approx_loss = (Lbar_pos + Lbar_neg) / 2
    return seed, scalar, approx_loss

@torch.no_grad()
def apply_update(model, seed, scalar, lr, mode="raw"):
    if mode == "sign":
        step = np.sign(scalar)
    elif mode == "norm":
        step = np.clip(scalar, -1.0, 1.0)  # single scalar -> just clip as fallback
    else:
        step = scalar
    if step == 0:
        return
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            z = torch.randn_like(p)
            p.add_(z, alpha=-lr * step)

@torch.no_grad()
def apply_update_adam(model, seed, scalar, adam_state, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    beta1, beta2 = betas
    torch.manual_seed(seed)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        z = torch.randn_like(p)
        g = scalar * z
        st = adam_state[id(p)]
        st["m"].mul_(beta1).add_(g, alpha=1 - beta1)
        st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
        m_hat = st["m"] / (1 - beta1 ** t)
        v_hat = st["v"] / (1 - beta2 ** t)
        p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

def get_node_batches(node_iters):
    batches = []
    for i, dl in enumerate(node_loaders):
        try:
            x, y = next(node_iters[i])
        except StopIteration:
            node_iters[i] = iter(dl)
            x, y = next(node_iters[i])
        batches.append((x.to(device), y.to(device)))
    return batches

# ---------------- training loops ----------------
def train_zo(n_rounds, lr, mu, mode="sign", log_every=50, quiet=False):
    model = make_model()
    node_iters = [iter(dl) for dl in node_loaders]
    roll = deque(maxlen=100)
    log = []
    pbar = tqdm(range(n_rounds), desc=f"ZO-{mode} lr={lr} mu={mu}", disable=quiet)
    for r in pbar:
        batches = get_node_batches(node_iters)
        seed, scalar, approx_loss = zo_round_shared_seed(model, batches, mu=mu)
        apply_update(model, seed, scalar, lr, mode=mode)
        roll.append(approx_loss)
        if not quiet:
            pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")
        if r % log_every == 0 or r == n_rounds - 1:
            log.append({"round": r, "train_avg100": float(np.mean(roll)),
                        "test_loss": eval_test_loss(model), "test_acc": eval_test_acc(model)})
    return model, log

def train_zo_adam(n_rounds, lr, mu, betas=(0.9, 0.999), log_every=50, quiet=False):
    model = make_model()
    adam_state = {id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                  for p in model.parameters() if p.requires_grad}
    node_iters = [iter(dl) for dl in node_loaders]
    roll = deque(maxlen=100)
    log = []
    pbar = tqdm(range(1, n_rounds + 1), desc=f"ZO-Adam lr={lr} mu={mu}", disable=quiet)
    for r in pbar:
        batches = get_node_batches(node_iters)
        seed, scalar, approx_loss = zo_round_shared_seed(model, batches, mu=mu)
        apply_update_adam(model, seed, scalar, adam_state, t=r, lr=lr, betas=betas)
        roll.append(approx_loss)
        if not quiet:
            pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")
        if r % log_every == 0 or r == n_rounds:
            log.append({"round": r, "train_avg100": float(np.mean(roll)),
                        "test_loss": eval_test_loss(model), "test_acc": eval_test_acc(model)})
    return model, log

def train_fo(n_rounds, lr, log_every=50, quiet=False):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    node_iters = [iter(dl) for dl in node_loaders]
    roll = deque(maxlen=100)
    log = []
    pbar = tqdm(range(n_rounds), desc=f"FO lr={lr}", disable=quiet)
    for r in pbar:
        opt.zero_grad()
        losses = []
        for x, y in get_node_batches(node_iters):
            loss = loss_fn(model(x), y) / N_NODES
            loss.backward()
            losses.append(loss.item() * N_NODES)
        opt.step()
        roll.append(float(np.mean(losses)))
        if not quiet:
            pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")
        if r % log_every == 0 or r == n_rounds - 1:
            log.append({"round": r, "train_avg100": float(np.mean(roll)),
                        "test_loss": eval_test_loss(model), "test_acc": eval_test_acc(model)})
    return model, log

# ---------------- quick lr/mu grid sweep ----------------
def sweep(mode="sign", n_rounds=500, lrs=(1e-1, 1e-2, 1e-3), mus=(1e-1, 1e-2, 1e-3)):
    """mode: 'sign' | 'norm' | 'raw' | 'adam'. quiet=True로 개별 progress bar 끄고 결과만 표."""
    results = []
    combos = list(itertools.product(lrs, mus))
    for lr, mu in tqdm(combos, desc=f"sweep({mode})"):
        try:
            if mode == "adam":
                model, log = train_zo_adam(n_rounds, lr=lr, mu=mu, log_every=n_rounds - 1, quiet=True)
            else:
                model, log = train_zo(n_rounds, lr=lr, mu=mu, mode=mode, log_every=n_rounds - 1, quiet=True)
            final = log[-1]
            status = "OK"
            if not np.isfinite(final["test_loss"]) or final["test_loss"] > 10:
                status = "DIVERGED"
        except Exception as e:
            final = {"test_loss": float("nan"), "test_acc": 0.0}
            status = f"ERROR: {e}"
        results.append({"lr": lr, "mu": mu, "mode": mode,
                        "test_loss": final["test_loss"], "test_acc": final["test_acc"], "status": status})

    results.sort(key=lambda r: r["test_loss"] if np.isfinite(r["test_loss"]) else 1e9)
    print(f"\n=== sweep results ({mode}, {n_rounds} rounds) ===")
    print(f"{'lr':>10} {'mu':>10} {'test_loss':>10} {'test_acc':>8}  status")
    for r in results:
        print(f"{r['lr']:>10.4g} {r['mu']:>10.4g} {r['test_loss']:>10.4f} {r['test_acc']:>8.3f}  {r['status']}")
    return results

def sweep_adam_beta(n_rounds=500, lrs=(1e-3,), mus=(1e-2,),
                     betas_list=((0.9, 0.999), (0.5, 0.999), (0.9, 0.99), (0.5, 0.9), (0.0, 0.999))):
    """lr, mu는 좁혀놓고 betas 축만 sweep. 넓게 하려면 lrs/mus에 여러개 넣으면 4차원 grid."""
    results = []
    combos = list(itertools.product(lrs, mus, betas_list))
    for lr, mu, betas in tqdm(combos, desc="sweep(adam-beta)"):
        try:
            model, log = train_zo_adam(n_rounds, lr=lr, mu=mu, betas=betas,
                                        log_every=n_rounds - 1, quiet=True)
            final = log[-1]
            status = "OK"
            if not np.isfinite(final["test_loss"]) or final["test_loss"] > 10:
                status = "DIVERGED"
        except Exception as e:
            final = {"test_loss": float("nan"), "test_acc": 0.0}
            status = f"ERROR: {e}"
        results.append({"lr": lr, "mu": mu, "beta1": betas[0], "beta2": betas[1],
                        "test_loss": final["test_loss"], "test_acc": final["test_acc"], "status": status})

    results.sort(key=lambda r: r["test_loss"] if np.isfinite(r["test_loss"]) else 1e9)
    print(f"\n=== adam beta sweep ({n_rounds} rounds) ===")
    print(f"{'lr':>10} {'mu':>10} {'b1':>6} {'b2':>7} {'test_loss':>10} {'test_acc':>8}  status")
    for r in results:
        print(f"{r['lr']:>10.4g} {r['mu']:>10.4g} {r['beta1']:>6.2f} {r['beta2']:>7.4f} "
              f"{r['test_loss']:>10.4f} {r['test_acc']:>8.3f}  {r['status']}")
    return results


if __name__ == "__main__":
    N_ROUNDS = 500

    print("=== FO baseline (sanity check) ===")
    train_fo(N_ROUNDS, lr=0.1)

    print("\n=== narrowed lr/mu sweep: ZO sign ===")
    sweep_sign = sweep(mode="sign", n_rounds=N_ROUNDS,
                        lrs=(3e-4, 1e-3, 3e-3), mus=(5e-3, 1e-2, 2e-2))

    print("\n=== narrowed lr/mu sweep: ZO-Adam (default betas) ===")
    sweep_adam = sweep(mode="adam", n_rounds=N_ROUNDS,
                        lrs=(3e-4, 1e-3, 3e-3), mus=(5e-3, 1e-2, 2e-2))

    print("\n=== Adam beta sweep (lr=1e-3, mu=1e-2 fixed from best above) ===")
    sweep_beta = sweep_adam_beta(n_rounds=N_ROUNDS, lrs=(1e-3,), mus=(1e-2,),
                                  betas_list=((0.9, 0.999), (0.5, 0.999), (0.9, 0.99),
                                              (0.5, 0.9), (0.0, 0.999), (0.9, 0.9999)))

    with open("mnist_sweep_results.json", "w") as f:
        json.dump({"sign": sweep_sign, "adam": sweep_adam, "adam_beta": sweep_beta}, f, indent=2)
    print("\nsaved to mnist_sweep_results.json")
    with open("mnist_sweep_results.json", "w") as f:
        json.dump({"sign": sweep_sign, "adam": sweep_adam, "adam_beta": sweep_beta}, f, indent=2)
    print("\nsaved to mnist_sweep_results.json")
