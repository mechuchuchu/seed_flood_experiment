"""
High-beta1 / low-lr / long-horizon ZO-Adam test.
Intuition: shared-seed SPSA scalar per round is noisy (single directional sample).
If beta1 is high enough, momentum accumulates the true direction across many rounds
faster than noise, even with small lr. Test that directly instead of a big grid.

Run: python zo_adam_momentum_test.py
"""
import time
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

import mnist_seedflood_sweep as base

CONFIGS = [
    # (lr, mu, beta1, beta2, n_rounds, label)
    (3e-3, 1e-2, 0.9,   0.999,  300,  "baseline (기존 best)"),
    (1e-3, 1e-2, 0.99,  0.999,  3000, "beta1=0.99, lr낮춤, 10x rounds"),
    (5e-4, 1e-2, 0.99,  0.999,  5000, "beta1=0.99, lr더낮춤, 더길게"),
    (1e-3, 1e-2, 0.999, 0.9999, 5000, "beta1=0.999(초강력 momentum)"),
    (3e-4, 1e-2, 0.999, 0.9999, 8000, "beta1=0.999 + lr최소 + 최장"),
]


def run(lr, mu, beta1, beta2, n_rounds, label):
    model = base.make_model()
    adam_state = {id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                  for p in model.parameters() if p.requires_grad}
    node_iters = [iter(dl) for dl in base.node_loaders]
    roll = deque(maxlen=100)
    log = []
    t0 = time.time()
    pbar = tqdm(range(1, n_rounds + 1), desc=label)
    for r in pbar:
        batches = base.get_node_batches(node_iters)
        seed, scalar, approx_loss = base.zo_round_shared_seed(model, batches, mu=mu)
        base.apply_update_adam(model, seed, scalar, adam_state, t=r, lr=lr, betas=(beta1, beta2))
        roll.append(approx_loss)
        pbar.set_postfix(avg100=f"{np.mean(roll):.4f}")
        if r % max(1, n_rounds // 10) == 0 or r == n_rounds:
            log.append({"round": r, "train_avg100": float(np.mean(roll)),
                        "test_loss": base.eval_test_loss(model), "test_acc": base.eval_test_acc(model)})
    elapsed = time.time() - t0
    final = log[-1]
    print(f"  -> final test_loss={final['test_loss']:.4f} test_acc={final['test_acc']:.4f} ({elapsed:.1f}s)")
    return log


if __name__ == "__main__":
    results = {}
    for lr, mu, beta1, beta2, n_rounds, label in CONFIGS:
        print(f"\n=== {label} (lr={lr}, mu={mu}, beta1={beta1}, beta2={beta2}, rounds={n_rounds}) ===")
        results[label] = run(lr, mu, beta1, beta2, n_rounds, label)

    print("\n=== summary ===")
    print(f"{'label':<40} {'final_test_loss':>16} {'final_test_acc':>16}")
    for label, log in results.items():
        final = log[-1]
        print(f"{label:<40} {final['test_loss']:>16.4f} {final['test_acc']:>16.4f}")
