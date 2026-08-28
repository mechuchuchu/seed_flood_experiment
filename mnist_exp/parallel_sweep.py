"""
Massive parallel lr/mu/beta sweep for MNIST SeedFlood using multiprocessing.
Reuses model/data/train functions from mnist_seedflood_sweep.py (must be in same dir).

Key design:
- fork start method (default on Linux/Mac): dataset + node_loaders loaded ONCE in parent,
  child processes inherit via copy-on-write -> no redundant MNIST reloading per worker.
- each worker pins torch to 1 thread to avoid oversubscription (N_cores workers x N_threads each = chaos otherwise).
- imap_unordered + tqdm for live progress across thousands of combos.

Run: python parallel_sweep.py
"""
import os
import time
import json
import itertools
import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

# reuse everything: model, data loaders, train_zo/train_zo_adam/train_fo, eval fns
import mnist_seedflood_sweep as base


def _worker_init():
    # 워커당 1 thread로 고정 -> N_proc x N_thread 오버섭스크립션 방지
    torch.set_num_threads(1)


def run_one_combo(params):
    mode = params["mode"]
    lr = params["lr"]
    mu = params["mu"]
    betas = params.get("betas", (0.9, 0.999))
    n_rounds = params["n_rounds"]

    torch.manual_seed(hash((mode, lr, mu, betas)) % (2**31))  # combo마다 다른 init seed

    try:
        if mode == "adam":
            model, log = base.train_zo_adam(n_rounds, lr=lr, mu=mu, betas=betas,
                                             log_every=n_rounds - 1, quiet=True)
        elif mode == "fo":
            model, log = base.train_fo(n_rounds, lr=lr, log_every=n_rounds - 1, quiet=True)
        else:
            model, log = base.train_zo(n_rounds, lr=lr, mu=mu, mode=mode,
                                        log_every=n_rounds - 1, quiet=True)
        final = log[-1]
        status = "OK"
        if not np.isfinite(final["test_loss"]) or final["test_loss"] > 10:
            status = "DIVERGED"
        return {**params, "betas": betas, "test_loss": final["test_loss"],
                "test_acc": final["test_acc"], "status": status}
    except Exception as e:
        return {**params, "betas": betas, "test_loss": float("nan"), "test_acc": 0.0,
                "status": f"ERROR: {type(e).__name__}: {e}"}


def build_grid(n_rounds=300):
    combos = []
    lrs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    mus = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    betas_list = [(0.9, 0.999), (0.5, 0.999), (0.9, 0.99), (0.5, 0.9), (0.0, 0.999), (0.9, 0.9999)]

    for lr, mu in itertools.product(lrs, mus):
        combos.append({"mode": "sign", "lr": lr, "mu": mu, "n_rounds": n_rounds})
        combos.append({"mode": "raw", "lr": lr, "mu": mu, "n_rounds": n_rounds})

    for lr, mu, betas in itertools.product(lrs, mus, betas_list):
        combos.append({"mode": "adam", "lr": lr, "mu": mu, "betas": betas, "n_rounds": n_rounds})

    for lr in lrs:
        combos.append({"mode": "fo", "lr": lr, "mu": None, "n_rounds": n_rounds})

    return combos


def main():
    n_rounds = 300  # 조합 수가 많으니 라운드는 줄여서 1차 스크리닝, 유망한 애들만 나중에 500~1000으로 재검증
    n_workers = max(1, os.cpu_count() - 1)

    combos = build_grid(n_rounds=n_rounds)
    print(f"total combos: {len(combos)}, workers: {n_workers}, n_rounds each: {n_rounds}")

    t0 = time.time()
    results = []
    with mp.get_context("fork").Pool(processes=n_workers, initializer=_worker_init) as pool:
        for res in tqdm(pool.imap_unordered(run_one_combo, combos), total=len(combos)):
            results.append(res)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({elapsed/len(combos):.2f}s/combo avg)")

    results.sort(key=lambda r: r["test_loss"] if np.isfinite(r["test_loss"]) else 1e9)

    with open("parallel_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== top 20 ===")
    print(f"{'mode':>6} {'lr':>10} {'mu':>10} {'betas':>16} {'test_loss':>10} {'test_acc':>8}  status")
    for r in results[:20]:
        mu_str = f"{r['mu']:.4g}" if r["mu"] is not None else "-"
        betas_str = str(r.get("betas", "-"))
        print(f"{r['mode']:>6} {r['lr']:>10.4g} {mu_str:>10} {betas_str:>16} "
              f"{r['test_loss']:>10.4f} {r['test_acc']:>8.3f}  {r['status']}")

    n_diverged = sum(1 for r in results if r["status"] != "OK")
    print(f"\ndiverged/error: {n_diverged}/{len(results)}")
    print("saved full results to parallel_sweep_results.json")


if __name__ == "__main__":
    main()
