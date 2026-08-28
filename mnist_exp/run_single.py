"""
Single hyperparameter combo runner - designed to be invoked as a fresh OS process
(via bash xargs/GNU parallel), so no fork/spawn/CUDA-context sharing issues at all.
Each invocation is fully isolated; a crash only kills that one process.

Usage:
    python run_single.py --mode sign --lr 1e-3 --mu 1e-2 --n_rounds 500 \
        --out results/sign_lr1e-3_mu1e-2.json

    python run_single.py --mode adam --lr 1e-3 --mu 1e-2 --beta1 0.9 --beta2 0.999 \
        --n_rounds 500 --out results/adam_....json

    python run_single.py --mode fo --lr 1e-1 --n_rounds 500 --out results/fo_lr1e-1.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["sign", "raw", "norm", "adam", "fo"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--mu", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--n_rounds", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", required=True, help="output json path for this combo's result")
    return p


def main():
    args = build_parser().parse_args()

    # torch import 순서/device 설정은 각 프로세스가 독립이라 여기서 자유롭게 결정
    torch.set_num_threads(1)  # bash에서 -P N개 프로세스 띄울 거라 프로세스당 1 thread로 제한
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" 대신 -1: NVML UUID hex 파싱 버그 회피

    import mnist_seedflood_sweep as base  # import 시점에 데이터 로드 (이 프로세스 안에서만)

    torch.manual_seed(args.seed)

    t0 = time.time()
    status = "OK"
    try:
        if args.mode == "adam":
            model, log = base.train_zo_adam(args.n_rounds, lr=args.lr, mu=args.mu,
                                             betas=(args.beta1, args.beta2),
                                             log_every=max(1, args.n_rounds - 1), quiet=True)
        elif args.mode == "fo":
            model, log = base.train_fo(args.n_rounds, lr=args.lr,
                                        log_every=max(1, args.n_rounds - 1), quiet=True)
        else:
            model, log = base.train_zo(args.n_rounds, lr=args.lr, mu=args.mu, mode=args.mode,
                                        log_every=max(1, args.n_rounds - 1), quiet=True)
        final = log[-1]
        if not np.isfinite(final["test_loss"]) or final["test_loss"] > 10:
            status = "DIVERGED"
    except Exception as e:
        final = {"test_loss": float("nan"), "test_acc": 0.0}
        status = f"ERROR: {type(e).__name__}: {e}"

    result = {
        "mode": args.mode, "lr": args.lr, "mu": args.mu,
        "beta1": args.beta1, "beta2": args.beta2, "n_rounds": args.n_rounds,
        "seed": args.seed, "test_loss": final["test_loss"], "test_acc": final["test_acc"],
        "status": status, "elapsed_sec": round(time.time() - t0, 2),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)

    print(json.dumps(result))  # stdout으로도 찍어서 bash 로그에 남게


if __name__ == "__main__":
    sys.exit(main())
