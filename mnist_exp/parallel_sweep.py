"""
Massive parallel lr/mu/beta sweep for MNIST SeedFlood using multiprocessing.
Reuses model/data/train functions from mnist_seedflood_sweep.py (must be in same dir).

Key design:
- spawn start method: 각 worker가 깨끗한 인터프리터에서 base를 재import (MNIST도 worker당 1회 로드).
  fork+COW가 더 쌌지만 fork된 child에서 glibc heap corruption이 간헐 재발해서 포기.
- each worker pins torch to 1 thread to avoid oversubscription (N_cores workers x N_threads each = chaos otherwise).
- imap_unordered + per-result timeout: worker crash가 나도 sweep이 hang하지 않고
  유실 combo를 parallel_sweep_missing.json으로 보고.

Run: python parallel_sweep.py
"""
import os

# fork + CUDA는 양립 불가 (부모에서 CUDA context 초기화되면 fork된 child에서 재초기화 에러).
# 이 sweep은 tiny MLP × 수백 combo를 CPU worker로 병렬화하는 설계라 (COW로 데이터 공유,
# worker당 1 thread), torch import 전에 GPU를 숨겨서 base가 device="cpu"로 뜨게 함.
# GPU로 돌리고 싶으면 이 줄을 지우고 spawn + worker당 device 할당 방식으로 바꿔야 함 (하단 주석 참고).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# fork-safety 핵심: 부모가 torch import 전에 thread pool을 1로 강제.
# 부모에서 libgomp/MKL이 멀티스레드 팀을 띄운 상태로 fork하면 child가 물려받은
# allocator/스레드 상태가 깨진 채 시작 → "corrupted size vs. prev_size" 같은
# heap corruption이 간헐적으로 터짐. child에서 set_num_threads(1) 하는 건
# 이미 늦음 (fork 시점에 부모가 single-thread여야 함).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

# [GPU로 돌리고 싶을 때 (참고)]
# 1) 위의 CUDA_VISIBLE_DEVICES="" 줄 제거
# 2) mp.get_context("spawn") 사용 — 단, spawn은 child가 base를 재import하므로
#    MNIST가 worker마다 다시 로드됨 (COW 공유 무효). MNIST는 작아서 감수 가능.
# 3) worker 수를 GPU당 1~2개로 제한하고 initializer에서
#    torch.cuda.set_device(worker_rank % n_gpus) 식으로 할당.
#    tiny MLP는 GPU 한 개에 worker 여러 개 붙이면 context 경합으로 오히려 느려짐.


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
    # worker가 죽으면 (heap corruption 등) 그 task의 result는 영원히 안 옴 →
    # 기본 imap_unordered는 여기서 무한 대기 (98%에서 util 0으로 멈추는 증상).
    # IMapIterator.next(timeout=...)로 stall을 감지하고 수집된 것만 들고 빠져나옴.
    RESULT_TIMEOUT = 60  # 초. 가장 느린 combo 하나보다 넉넉하게.
    # spawn: fork의 COW 공유를 포기하는 대신 각 child가 깨끗한 인터프리터에서 시작.
    # (부모 single-thread화로도 corruption이 재발해서 fork 포기.
    #  비용: worker당 torch import + MNIST 재로드 ~수 초, 시작할 때 한 번뿐이라 감수.)
    with mp.get_context("spawn").Pool(processes=n_workers, initializer=_worker_init) as pool:
        it = pool.imap_unordered(run_one_combo, combos)
        with tqdm(total=len(combos)) as pbar:
            while True:
                try:
                    res = it.next(timeout=RESULT_TIMEOUT)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    print(f"\n[!] {RESULT_TIMEOUT}s 동안 결과 없음 — worker crash로 판단, "
                          f"수집된 {len(results)}/{len(combos)}개만 저장하고 종료")
                    break
                results.append(res)
                pbar.update(1)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({elapsed/max(1,len(results)):.2f}s/combo avg)")

    # 유실된 combo 리포트 (죽은 worker가 들고 있던 것들) → 나중에 단독 재실행용
    def _key(c):
        return (c["mode"], c["lr"], c["mu"], tuple(c.get("betas", (0.9, 0.999))) if c["mu"] is not None else None)
    done_keys = {_key(r) for r in results}
    missing = [c for c in combos if _key(c) not in done_keys]
    if missing:
        print(f"[!] 유실 combo {len(missing)}개:")
        for c in missing:
            print(f"    {c}")
        with open("parallel_sweep_missing.json", "w") as f:
            json.dump(missing, f, indent=2, default=str)

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
