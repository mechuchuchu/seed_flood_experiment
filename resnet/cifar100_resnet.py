"""
CIFAR-100 (HuggingFace uoft-cs/cifar100) + CIFAR-style ResNet
FO baseline + shared-seed ZO (SPSA) — sign / Adam 모드

- torchvision 서버 이슈 우회: datasets 라이브러리로 로드
- 전체 데이터를 uint8 텐서로 RAM(또는 GPU)에 상주, DataLoader 없이 인덱스 샘플링
- shared-seed SPSA: 라운드당 seed 1개 → 전 노드 같은 z 방향, loss 노드 평균 →
  scalar = (L+ − L−)/(2μ), pseudo_grad = scalar·z  (mnist_seedflood_sweep.py와 동일 구조)
- z는 저장하지 않고 seed로 3회 재생성 (perturb +μ / −2μ / 복원 +μ) → 메모리 O(params) 추가분 없음
  (단 zo_adam은 m, v 버퍼로 2x params 추가)
- 모든 결과는 json 하나에 저장: args(hyperparams) + round별 시계열 + final
  → aggregate_results.py에 그대로 물릴 수 있는 flat 구조

메모리: train 50000x3x32x32 uint8 ≈ 154MB, test ≈ 31MB (--gpu_resident로 VRAM 상주 가능)

주의(BN vs ZO): BatchNorm은 ±μ 두 번의 forward마다 running stats를 갱신해서 ZO의
loss 차이 측정을 오염시킴 → --norm auto는 zo 모드에서 GroupNorm을 선택함.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Data: HF uoft-cs/cifar100 → RAM tensors
# ----------------------------------------------------------------------------

CIFAR100_MEAN = torch.tensor([0.5071, 0.4865, 0.4409])
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762])


def load_cifar100_ram(label_key: str = "fine_label", storage_device: str = "cpu"):
    """HF에서 CIFAR-100 받아서 전체를 uint8 텐서로 메모리에 올림.

    Returns:
        (x_train, y_train, x_test, y_test)
        x_*: (N, 3, 32, 32) uint8, y_*: (N,) long
    """
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar100")  # 첫 실행만 다운로드, 이후 HF cache

    def split_to_tensors(split):
        imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in split["img"]])
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # (N,3,32,32)
        y = torch.tensor(split[label_key], dtype=torch.long)
        return x.to(storage_device), y.to(storage_device)

    t0 = time.time()
    x_train, y_train = split_to_tensors(ds["train"])
    x_test, y_test = split_to_tensors(ds["test"])
    print(
        f"[data] train {tuple(x_train.shape)} test {tuple(x_test.shape)} "
        f"on {storage_device}, loaded in {time.time() - t0:.1f}s"
    )
    return x_train, y_train, x_test, y_test


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,32,32) → normalized float32. 배치 단위로 GPU에서 호출."""
    mean = CIFAR100_MEAN.to(x_uint8.device).view(1, 3, 1, 1)
    std = CIFAR100_STD.to(x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def augment_batch(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """벡터화된 random crop(+pad) & horizontal flip. float 입력, 배치 전체 GPU 처리."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (oy.view(b, 1) + torch.arange(h, device=x.device)).view(b, 1, h, 1)
    cols = (ox.view(b, 1) + torch.arange(w, device=x.device)).view(b, 1, 1, w)
    xc = xp.gather(2, rows.expand(b, c, h, xp.size(3)))
    xc = xc.gather(3, cols.expand(b, c, h, w))
    flip = torch.rand(b, device=x.device) < 0.5
    xc[flip] = torch.flip(xc[flip], dims=[3])
    return xc


def get_batch(x, y, batch_size, device, indices=None, augment=False, generator=None):
    """RAM 상주 텐서에서 랜덤 배치 하나. indices 주면 그 부분집합(=노드 파티션)에서만 샘플."""
    if indices is None:
        idx = torch.randint(0, x.size(0), (batch_size,), generator=generator)
    else:
        pick = torch.randint(0, indices.numel(), (batch_size,), generator=generator)
        idx = indices[pick]
    xb = x[idx].to(device, non_blocking=True)
    yb = y[idx].to(device, non_blocking=True)
    xb = normalize(xb)
    if augment:
        xb = augment_batch(xb)
    return xb, yb


def build_node_indices(n_total: int, n_nodes: int, seed: int = 0):
    """SeedFlood용: train set을 노드별로 disjoint하게 분할한 index 리스트."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return list(torch.chunk(perm, n_nodes))


@torch.no_grad()
def evaluate(model, x_test, y_test, device, batch_size=1000):
    was_training = model.training
    model.eval()
    total_loss, correct = 0.0, 0
    n = x_test.size(0)
    for i in range(0, n, batch_size):
        xb = normalize(x_test[i : i + batch_size].to(device))
        yb = y_test[i : i + batch_size].to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    if was_training:
        model.train()
    return total_loss / n, correct / n


# ----------------------------------------------------------------------------
# CIFAR-style ResNet (3x3 stem, no maxpool), norm 선택 가능
# ----------------------------------------------------------------------------


def make_norm(kind: str, ch: int):
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(8, ch)  # 8은 16/32/64/.../512 전부 나누어떨어짐
    raise ValueError(kind)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, norm="batch"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.n1 = make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.n2 = make_norm(norm, out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                make_norm(norm, out_ch),
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks, widths, num_classes=100, norm="batch"):
        super().__init__()
        self.in_ch = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.n1 = make_norm(norm, widths[0])
        layers = []
        for i, (nb, w) in enumerate(zip(num_blocks, widths)):
            stride = 1 if i == 0 else 2
            layers.append(self._make_layer(w, nb, stride, norm))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(widths[-1], num_classes)

    def _make_layer(self, out_ch, n_blocks, stride, norm):
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_ch, out_ch, s, norm))
            self.in_ch = out_ch
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20(num_classes=100, norm="batch"):
    """~0.28M params — ZO 스크리닝용으로 현실적인 크기"""
    return ResNet([3, 3, 3], [16, 32, 64], num_classes, norm)


def resnet18(num_classes=100, norm="batch"):
    """~11.2M params — FO baseline / 나중 스케일업용"""
    return ResNet([2, 2, 2, 2], [64, 128, 256, 512], num_classes, norm)


MODELS = {"resnet20": resnet20, "resnet18": resnet18}


# ----------------------------------------------------------------------------
# Logging: hyperparams + 시계열 → json
# ----------------------------------------------------------------------------


class RunLogger:
    def __init__(self, args, n_params):
        self.record = {
            "args": vars(args),          # 하이퍼파라미터 전부 (lr, mu, beta1, beta2, ...)
            "n_params": n_params,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "history": [],               # round별 {round, train_avg, test_loss, test_acc, elapsed}
            "final": None,
        }
        self.out = args.out
        os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)

    def log(self, rnd, train_avg, test_loss, test_acc, elapsed, lr=None):
        self.record["history"].append({
            "round": rnd,
            "train_avg": round(float(train_avg), 6),
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 6),
            "lr": round(float(lr), 8) if lr is not None else None,
            "elapsed": round(elapsed, 1),
        })
        self._dump()  # 매 eval마다 저장 → 중간에 죽어도 로그 살아있음

    def finalize(self, test_loss, test_acc, elapsed):
        self.record["final"] = {
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 6),
            "total_sec": round(elapsed, 1),
        }
        self._dump()
        print(f"[log] saved → {self.out}")

    def _dump(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record, f, indent=2)
        os.replace(tmp, self.out)


# ----------------------------------------------------------------------------
# LR schedule: linear warmup → constant | cosine
# ----------------------------------------------------------------------------


def lr_at(rnd, args):
    """rnd(1-indexed)에서의 lr. warmup 구간은 0→lr 선형 증가."""
    if args.warmup_rounds > 0 and rnd <= args.warmup_rounds:
        return args.lr * rnd / args.warmup_rounds
    if args.lr_schedule == "cosine":
        t = (rnd - args.warmup_rounds) / max(1, args.n_rounds - args.warmup_rounds)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))
    return args.lr  # constant


# ----------------------------------------------------------------------------
# FO baseline
# ----------------------------------------------------------------------------


def train_fo(model, data, device, args, logger):
    x_train, y_train, x_test, y_test = data
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)

    running, t0 = [], time.time()
    tl = ta = float("nan")
    for step in range(1, args.n_rounds + 1):
        cur_lr = lr_at(step, args)
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        xb, yb = get_batch(x_train, y_train, args.batch_size, device,
                           augment=args.use_augment)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % args.eval_every == 0 or step == args.n_rounds:
            tl, ta = evaluate(model, x_test, y_test, device)
            el = time.time() - t0
            logger.log(step, np.mean(running), tl, ta, el, lr=cur_lr)
            print(f"step {step:6d} | train_avg {np.mean(running):.4f} | "
                  f"test_loss {tl:.4f} | test_acc {ta:.4f} | {el:.0f}s")
            running = []
    logger.finalize(tl, ta, time.time() - t0)


# ----------------------------------------------------------------------------
# Shared-seed ZO (SPSA)
# ----------------------------------------------------------------------------


@torch.no_grad()
def zo_perturb_(params, seed, scale, device):
    """seed로 z를 재생성하며 in-place perturb. z 저장 안 함."""
    g = torch.Generator(device=device).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=g, device=device, dtype=p.dtype)
        p.add_(scale * z)


@torch.no_grad()
def zo_loss(model, batches):
    """노드별 로컬 배치 loss의 평균 (SeedFlood의 서버측 평균에 해당)."""
    total = 0.0
    for xb, yb in batches:
        total += F.cross_entropy(model(xb), yb).item()
    return total / len(batches)


def train_zo(model, data, device, args, logger):
    x_train, y_train, x_test, y_test = data
    model.train()  # (norm=group이면 train/eval 동작 동일, dropout 없음)
    params = list(model.parameters())
    node_indices = build_node_indices(x_train.size(0), args.n_nodes, seed=args.data_seed)

    if args.mode == "zo_adam":
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]

    running, t0 = [], time.time()
    tl = ta = float("nan")
    for rnd in range(1, args.n_rounds + 1):
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())  # 라운드당 seed 1개 broadcast

        # 각 노드가 로컬 배치 하나씩 샘플 (± 두 평가에 같은 배치 재사용)
        batches = [
            get_batch(x_train, y_train, args.batch_size, device,
                      indices=ni, augment=args.use_augment)
            for ni in node_indices
        ]

        zo_perturb_(params, seed, +args.mu, device)
        l_plus = zo_loss(model, batches)
        zo_perturb_(params, seed, -2 * args.mu, device)
        l_minus = zo_loss(model, batches)
        zo_perturb_(params, seed, +args.mu, device)  # θ 복원

        scalar = (l_plus - l_minus) / (2 * args.mu)
        running.append((l_plus + l_minus) / 2)
        cur_lr = lr_at(rnd, args)

        # update: 같은 seed로 z 재생성 → pseudo_grad = scalar·z
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for i, p in enumerate(params):
                if args.zo_weight_decay > 0:
                    p.mul_(1 - cur_lr * args.zo_weight_decay)  # decoupled (AdamW식)
                z = torch.randn(p.shape, generator=g, device=device, dtype=p.dtype)
                grad = scalar * z
                if args.mode == "zo_sign":
                    p.sub_(cur_lr * grad.sign())
                else:  # zo_adam
                    m[i].mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
                    v[i].mul_(args.beta2).addcmul_(grad, grad, value=1 - args.beta2)
                    m_hat = m[i] / (1 - args.beta1 ** rnd)
                    v_hat = v[i] / (1 - args.beta2 ** rnd)
                    p.sub_(cur_lr * m_hat / (v_hat.sqrt() + args.eps))

        if rnd % args.eval_every == 0 or rnd == args.n_rounds:
            tl, ta = evaluate(model, x_test, y_test, device)
            el = time.time() - t0
            logger.log(rnd, np.mean(running), tl, ta, el, lr=cur_lr)
            print(f"round {rnd:6d} | train_avg {np.mean(running):.4f} | "
                  f"test_loss {tl:.4f} | test_acc {ta:.4f} | {el:.0f}s")
            running = []
    logger.finalize(tl, ta, time.time() - t0)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def default_out(args):
    tag = (f"{args.mode}_{args.model}_lr{args.lr:g}_mu{args.mu:g}"
           f"_b1{args.beta1:g}_b2{args.beta2:g}_bs{args.batch_size}"
           f"_n{args.n_nodes}_r{args.n_rounds}")
    return os.path.join("results", tag + ".json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="fo", choices=["fo", "zo_sign", "zo_adam"])
    p.add_argument("--model", default="resnet20", choices=list(MODELS))
    p.add_argument("--norm", default="auto", choices=["auto", "batch", "group"],
                   help="auto: fo→batch, zo→group (BN running stats 오염 회피)")
    # 공통 hyperparams
    p.add_argument("--lr", type=float, default=None,
                   help="default: fo=0.1, zo_sign=1e-3, zo_adam=1e-3")
    p.add_argument("--warmup_rounds", type=int, default=0,
                   help="0→lr 선형 warmup 구간 길이 (0이면 warmup 없음)")
    p.add_argument("--lr_schedule", default="auto",
                   choices=["auto", "constant", "cosine"],
                   help="warmup 이후 스케줄. auto: fo=cosine, zo=constant "
                        "(기존 MNIST sweep과 조건 맞추기 위함)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_rounds", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=5e-4, help="fo에서만 사용")
    # ZO hyperparams
    p.add_argument("--mu", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.99)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--zo_weight_decay", type=float, default=0.0,
                   help="ZO용 decoupled weight decay (AdamW 방식: p *= 1 - lr*wd). "
                        "pseudo_grad에 더하지 않으므로 m/v 통계를 오염시키지 않음")
    p.add_argument("--n_nodes", type=int, default=1)
    # misc
    p.add_argument("--augment", default="auto", choices=["auto", "on", "off"],
                   help="auto: fo=on, zo=off (aug 노이즈가 loss 차이 측정을 오염시킴)")
    p.add_argument("--label_key", default="fine_label",
                   choices=["fine_label", "coarse_label"])
    p.add_argument("--gpu_resident", action="store_true",
                   help="데이터셋 전체를 VRAM에 상주 (H2D copy 제거)")
    p.add_argument("--data_seed", type=int, default=0, help="노드 파티션 seed")
    p.add_argument("--torch_seed", type=int, default=None, help="모델 init 재현용")
    p.add_argument("--out", default=None, help="결과 json 경로 (default: 자동 이름)")
    args = p.parse_args()

    is_zo = args.mode.startswith("zo")
    if args.lr is None:
        args.lr = 0.1 if args.mode == "fo" else 1e-3
    if args.norm == "auto":
        args.norm = "group" if is_zo else "batch"
    if args.lr_schedule == "auto":
        args.lr_schedule = "constant" if is_zo else "cosine"
    if is_zo and args.norm == "batch":
        print("[warn] zo + BatchNorm: ±μ forward마다 running stats가 갱신되어 "
              "측정이 오염될 수 있음 (--norm group 권장)")
    args.use_augment = (args.mode == "fo") if args.augment == "auto" else (args.augment == "on")
    if args.out is None:
        args.out = default_out(args)

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage = device if (args.gpu_resident and device == "cuda") else "cpu"

    data = load_cifar100_ram(label_key=args.label_key, storage_device=storage)
    num_classes = 100 if args.label_key == "fine_label" else 20
    model = MODELS[args.model](num_classes=num_classes, norm=args.norm).to(device)
    n_params = sum(par.numel() for par in model.parameters())
    print(f"[model] {args.model} ({args.norm} norm): {n_params / 1e6:.2f}M params, "
          f"mode={args.mode}, device={device}")
    print(f"[out] {args.out}")

    logger = RunLogger(args, n_params)
    if args.mode == "fo":
        train_fo(model, data, device, args, logger)
    else:
        train_zo(model, data, device, args, logger)
