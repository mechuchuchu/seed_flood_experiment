"""
CIFAR-100 (HuggingFace uoft-cs/cifar100) + CIFAR-style ResNet
FO baseline + shared-seed ZO (SPSA) — sign / Adam 모드

- torchvision 서버 이슈 우회: datasets 라이브러리로 로드
- 전체 데이터를 uint8 텐서로 RAM(또는 GPU)에 상주, DataLoader 없이 인덱스 샘플링
- shared-seed SPSA: 라운드당 seed 1개 → 전 노드 같은 z 방향, loss 노드 평균 →
  scalar = (L+ − L−)/(2μ), pseudo_grad = scalar·z  (mnist_seedflood_sweep.py와 동일 구조)
- z는 저장하지 않고 seed로 재생성 (±μ perturb 2회 + update 1회), θ 복원은 스냅샷 copy
  (산술 복원은 bf16에서 rounding 잔차가 μ 스케일로 남음 — 아래 dtype 주의 참고)
  (zo_adam은 m, v 버퍼로 2x params 추가, fp32/fp64 master)
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


def get_batch(x, y, batch_size, device, indices=None, augment=False, generator=None,
              dtype=None):
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
    if dtype is not None:
        xb = xb.to(dtype)  # normalize/augment는 fp32에서 하고 마지막에 캐스팅
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
    dtype = next(model.parameters()).dtype
    total_loss, correct = 0.0, 0
    n = x_test.size(0)
    for i in range(0, n, batch_size):
        xb = normalize(x_test[i : i + batch_size].to(device)).to(dtype)
        yb = y_test[i : i + batch_size].to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits.float(), yb, reduction="sum").item()
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
# Parameter initialization
# ----------------------------------------------------------------------------


def init_model(model, scheme="default", gain=1.0, fc_scale=1.0, zero_init_residual=False):
    """Conv/Linear weight 초기화 스킴 적용.

    scheme:
      default        : PyTorch 기본 (Conv2d/Linear 모두 kaiming_uniform, a=√5
                       → 사실상 gain이 작은 uniform. 아래 kaiming_*와 다름)
      kaiming_normal : He normal, fan_out + relu (ResNet 논문 표준)
      kaiming_uniform: He uniform, fan_out + relu
      xavier_normal  : Glorot normal
      xavier_uniform : Glorot uniform
      orthogonal     : 직교 초기화 (fan-in 축 기준 reshape)

    gain: 위 스킴으로 초기화한 weight 전체에 곱하는 스케일.
          ZO에서 perturbation ‖μz‖=μ√d 대비 ‖θ‖ 비율을 조절하고 싶을 때 사용.
    fc_scale: 마지막 분류기(fc) weight에만 추가로 곱하는 스케일.
          작게 주면 초기 로짓이 작아져 softmax가 균등에 가까워지고, 크게 주면 반대.
    zero_init_residual: 각 BasicBlock의 두 번째 norm weight를 0으로 → 블록이 항등함수로
          시작 (Goyal et al. 2017). 깊은 net에서 초기 신호 전파 안정화.
    """
    norm_types = (nn.BatchNorm2d, nn.GroupNorm)
    if scheme != "default":
        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                w = mod.weight
                if scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
                elif scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(w, mode="fan_out", nonlinearity="relu")
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(w)
                elif scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(w)
                elif scheme == "orthogonal":
                    nn.init.orthogonal_(w)
                else:
                    raise ValueError(f"unknown init scheme: {scheme}")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        for mod in model.modules():  # norm affine은 스킴과 무관하게 표준값 유지
            if isinstance(mod, norm_types) and mod.weight is not None:
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    if gain != 1.0:
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, (nn.Conv2d, nn.Linear)):
                    mod.weight.mul_(gain)

    if fc_scale != 1.0:
        with torch.no_grad():
            model.fc.weight.mul_(fc_scale)
            if model.fc.bias is not None:
                model.fc.bias.mul_(fc_scale)

    if zero_init_residual:
        for mod in model.modules():
            if isinstance(mod, BasicBlock) and mod.n2.weight is not None:
                nn.init.zeros_(mod.n2.weight)

    return model


def param_stats(model):
    """초기화 결과 요약 (ZO에서 ‖θ‖ 대비 perturbation 크기 감 잡기용)."""
    with torch.no_grad():
        sq = sum((p.double() ** 2).sum() for p in model.parameters())
        d = sum(p.numel() for p in model.parameters())
    return {"n_params": d, "theta_norm": float(sq.sqrt()), "sqrt_d": math.sqrt(d)}


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

    def log(self, rnd, train_avg, test_loss, test_acc, elapsed, lr=None, **extra):
        entry = {
            "round": rnd,
            "train_avg": round(float(train_avg), 6),
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 6),
            "lr": round(float(lr), 8) if lr is not None else None,
            "elapsed": round(elapsed, 1),
        }
        entry.update(extra)  # e.g. precision probe: zo_delta / zo_delta_fp64 / zo_delta_err
        self.record["history"].append(entry)
        self._dump()  # 매 eval마다 저장 → 중간에 죽어도 로그 살아있음

    def finalize(self, test_loss, test_acc, elapsed, aborted=None):
        self.record["final"] = {
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 6),
            "total_sec": round(elapsed, 1),
            "aborted": aborted,  # None이면 정상 완주, 문자열이면 조기 종료 사유
        }
        self._dump()
        print(f"[log] saved → {self.out}")

    def _dump(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record, f, indent=2)
        os.replace(tmp, self.out)


class CollapseWatch:
    """ZO 붕괴 조기 감지.

    관찰: 붕괴한 런에서는 |g^T z| (= |grad_term|/(2μ), grad_term 없으면 |delta|/(2μ))가
    초기 대비 자릿수로 떨어지고 loss가 ln(num_classes) 근처에 못박힌다.
    grid sweep에서 이런 조합에 20분씩 쓰는 걸 막기 위해 조기 종료한다.

    발동 조건 (min_round 이후, 전부 만족):
      - 최근 관측의 |g^T z| 중앙값이 초기 기준값의 1/ratio 이하
      - loss가 uniform baseline(ln C) 근처에서 개선 없음
    """

    def __init__(self, args, num_classes):
        self.enabled = args.early_abort
        self.min_round = args.abort_min_round
        self.ratio = args.abort_ratio
        self.uniform_loss = math.log(num_classes)
        self.baseline = None      # 초기 |g^T z| 기준값
        self.init_samples = []
        self.recent = []
        self.best_loss = float("inf")
        self.reason = None

    def update(self, rnd, signal, test_loss):
        """signal = |g^T z| 추정치. 반환 True면 중단."""
        if not self.enabled or signal is None:
            return False
        self.best_loss = min(self.best_loss, test_loss)
        if self.baseline is None:
            self.init_samples.append(signal)
            if len(self.init_samples) >= 3:
                self.baseline = float(np.median(self.init_samples))
            return False
        self.recent.append(signal)
        self.recent = self.recent[-3:]
        if rnd < self.min_round or len(self.recent) < 3:
            return False
        cur = float(np.median(self.recent))
        signal_dead = self.baseline > 0 and cur < self.baseline / self.ratio
        no_progress = self.best_loss > self.uniform_loss * 0.99
        if signal_dead and no_progress:
            self.reason = (f"signal collapse: |g^T z| {self.baseline:.3e} → {cur:.3e} "
                           f"(x{cur / self.baseline:.1e}), best_loss {self.best_loss:.4f} "
                           f"≥ uniform {self.uniform_loss:.4f}")
            return True
        return False


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
    dtype = next(model.parameters()).dtype
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)

    running, t0 = [], time.time()
    tl = ta = float("nan")
    for step in range(1, args.n_rounds + 1):
        cur_lr = lr_at(step, args)
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        xb, yb = get_batch(x_train, y_train, args.batch_size, device,
                           augment=args.use_augment, dtype=dtype)
        loss = F.cross_entropy(model(xb).float(), yb)
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
    logger.record["rounds_done"] = args.n_rounds
    logger._dump()


# ----------------------------------------------------------------------------
# Shared-seed ZO (SPSA)
# ----------------------------------------------------------------------------


def zo_gen_dtype(p):
    """z 생성용 dtype: 파라미터가 fp64면 fp64, 그 외(fp32/bf16)는 fp32.
    bf16에서 직접 randn을 뽑지 않는 이유: perturb와 update가 같은 z를 재생성해야
    하는데, 생성/스케일링을 fp32 경로로 고정해두면 dtype별 RNG 구현 차이에서 자유로움."""
    return torch.float64 if p.dtype == torch.float64 else torch.float32


@torch.no_grad()
def zo_perturb_(params, seed, scale, device):
    """seed로 z를 재생성하며 in-place perturb. z 저장 안 함."""
    g = torch.Generator(device=device).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=g, device=device, dtype=zo_gen_dtype(p))
        p.add_((scale * z).to(p.dtype))


@torch.no_grad()
def zo_loss(model, batches):
    """노드별 로컬 배치 loss의 평균 (SeedFlood의 서버측 평균에 해당).
    forward는 모델 dtype 그대로 두되(=dtype 실험의 대상), CE 집계는 fp32로."""
    total = 0.0
    for xb, yb in batches:
        total += F.cross_entropy(model(xb).float(), yb).item()
    return total / len(batches)


@torch.no_grad()
def precision_probe(model, params, theta, batches, seed, mu, device, bias3=False):
    """같은 θ / 같은 z / 같은 배치에서 L0, L±μ (옵션: L±μ/2)를 fp64로 재측정.

    - 호출 시점 조건: θ가 스냅샷(theta)으로 복원된 상태
    - z는 학습 경로와 동일하게 fp32에서 생성 후 fp64로 승격 (같은 방향 보장)
    - 반환 dict:
        delta  = L+ − L−            (SPSA 신호)
        dplus  = L+ − L0            (한쪽 차분)
        dminus = L0 − L−            (반대쪽 차분)
        curv   = dplus − dminus     (= L+ + L− − 2L0 ≈ μ²·zᵀHz, 선형근사 위반량)
      bias3=True면 추가로 (forward 2회 더):
        delta_half = μ/2에서의 (L+ − L−)
        bias3      = (delta − 2·delta_half) · 4/3
                     Richardson: ΔL(μ)=2μ·gᵀz + (μ³/3)·T[z,z,z] + O(μ⁵) 이므로
                     ΔL(μ) − 2ΔL(μ/2) = (μ³/3)(1 − 1/4)·T = (μ³/4)·T
                     → ×4/3 하면 delta에 실제로 실린 3차항 (μ³/3)·T 크기가 나옴
        grad_term  = delta − bias3  (1차항 추정치)
    """
    orig_dtype = next(model.parameters()).dtype
    model.double()
    params64 = list(model.parameters())
    theta64 = [t.to(torch.float64) for t in theta]
    batches64 = [(xb.to(torch.float64), yb) for xb, yb in batches]

    def set_perturbed(offset):
        g = torch.Generator(device=device).manual_seed(seed)
        for p_, t_ in zip(params64, theta64):
            z = torch.randn(p_.shape, generator=g, device=device, dtype=torch.float32)
            p_.data.copy_(t_ + offset * z.double())

    l_zero = zo_loss(model, batches64)  # params64는 아직 θ 그대로
    set_perturbed(+mu)
    l_plus = zo_loss(model, batches64)
    set_perturbed(-mu)
    l_minus = zo_loss(model, batches64)

    dplus, dminus = l_plus - l_zero, l_zero - l_minus
    out = {"delta": l_plus - l_minus, "dplus": dplus, "dminus": dminus,
           "curv": dplus - dminus}

    if bias3:
        set_perturbed(+mu / 2)
        lp_h = zo_loss(model, batches64)
        set_perturbed(-mu / 2)
        lm_h = zo_loss(model, batches64)
        delta_half = lp_h - lm_h
        b3 = (out["delta"] - 2 * delta_half) * (4.0 / 3.0)
        out.update({"delta_half": delta_half, "bias3": b3,
                    "grad_term": out["delta"] - b3})

    # 원상복구: dtype 되돌리고 θ 재주입
    model.to(orig_dtype)
    for p_, t_ in zip(model.parameters(), theta):
        p_.data.copy_(t_)
    return out


def train_zo(model, data, device, args, logger):
    x_train, y_train, x_test, y_test = data
    model.train()  # (norm=group이면 train/eval 동작 동일, dropout 없음)
    params = list(model.parameters())
    dtype = next(model.parameters()).dtype
    node_indices = build_node_indices(x_train.size(0), args.n_nodes, seed=args.data_seed)

    # optimizer state는 파라미터가 bf16이어도 fp32 master로 유지
    # (bf16은 mantissa 8bit라 beta1=0.999 같은 계수 자체가 표현이 안 되고,
    #  (1-beta2)=1e-4 스케일 누적이 바로 소실됨)
    if args.mode == "zo_adam":
        m = [torch.zeros(p.shape, device=device, dtype=zo_gen_dtype(p)) for p in params]
        v = [torch.zeros(p.shape, device=device, dtype=zo_gen_dtype(p)) for p in params]

    running, t0 = [], time.time()
    tl = ta = float("nan")
    watch = CollapseWatch(args, model.fc.out_features)
    sig_window = []  # 매 라운드 |g^T z| (probe 없어도 delta로 계산 가능)
    for rnd in range(1, args.n_rounds + 1):
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())  # 라운드당 seed 1개 broadcast

        # 각 노드가 로컬 배치 하나씩 샘플 (± 두 평가에 같은 배치 재사용)
        batches = [
            get_batch(x_train, y_train, args.batch_size, device,
                      indices=ni, augment=args.use_augment, dtype=dtype)
            for ni in node_indices
        ]

        # θ 스냅샷 복원 방식: 산술 복원(+μ→−2μ→+μ)은 bf16에서 rounding 잔차가
        # μ와 같은 스케일(~1e-2)로 남아 매 라운드 θ를 오염시킴 → copy로 정확 복원.
        # 두 평가 모두 같은 θ에서 ±quantize(μz)로 대칭 처리됨. 메모리 +1x params.
        theta = [p.detach().clone() for p in params]
        zo_perturb_(params, seed, +args.mu, device)
        l_plus = zo_loss(model, batches)
        for p_, t_ in zip(params, theta):
            p_.data.copy_(t_)
        zo_perturb_(params, seed, -args.mu, device)
        l_minus = zo_loss(model, batches)
        for p_, t_ in zip(params, theta):
            p_.data.copy_(t_)

        scalar = (l_plus - l_minus) / (2 * args.mu)
        running.append((l_plus + l_minus) / 2)
        sig_window.append(abs(scalar))  # |g^T z| 추정 (bias 미보정)
        cur_lr = lr_at(rnd, args)

        is_eval_round = (rnd % args.eval_every == 0 or rnd == args.n_rounds)
        probe = {}
        if args.curvature_check and is_eval_round:
            # 대칭성 검증: (L+ − L0) =? (L0 − L−). 차이 = L+ + L− − 2L0 ≈ μ²·zᵀHz
            # θ는 지금 복원된 상태이므로 무섭동 loss를 같은 배치로 한 번 더 측정
            l_zero = zo_loss(model, batches)
            dplus, dminus = l_plus - l_zero, l_zero - l_minus
            probe.update({
                "zo_dplus": float(f"{dplus:.6e}"),
                "zo_dminus": float(f"{dminus:.6e}"),
                "zo_curv": float(f"{dplus - dminus:.6e}"),
            })
        if args.precision_probe and is_eval_round:
            # θ 복원된 지금 시점에 같은 seed/배치로 fp64 재측정
            delta_run = l_plus - l_minus
            pr = precision_probe(model, params, theta, batches,
                                 seed, args.mu, device, bias3=args.bias3_probe)
            probe.update({
                "zo_delta": float(f"{delta_run:.6e}"),
                "zo_delta_fp64": float(f"{pr['delta']:.6e}"),
                "zo_delta_err": float(f"{delta_run - pr['delta']:.6e}"),
                "zo_curv_fp64": float(f"{pr['curv']:.6e}"),
            })
            if args.bias3_probe:
                probe.update({
                    "zo_bias3": float(f"{pr['bias3']:.6e}"),
                    "zo_grad_term": float(f"{pr['grad_term']:.6e}"),
                })

        # update: 같은 seed로 z 재생성 → pseudo_grad = scalar·z
        # (계산은 fp32/fp64 master에서, 파라미터 적용 시에만 p.dtype으로 캐스팅)
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for i, p in enumerate(params):
                if args.zo_weight_decay > 0:
                    p.mul_(1 - cur_lr * args.zo_weight_decay)  # decoupled (AdamW식)
                z = torch.randn(p.shape, generator=g, device=device,
                                dtype=zo_gen_dtype(p))
                grad = scalar * z
                if args.mode == "zo_sign":
                    p.sub_((cur_lr * grad.sign()).to(p.dtype))
                else:  # zo_adam
                    m[i].mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
                    v[i].mul_(args.beta2).addcmul_(grad, grad, value=1 - args.beta2)
                    m_hat = m[i] / (1 - args.beta1 ** rnd)
                    v_hat = v[i] / (1 - args.beta2 ** rnd)
                    p.sub_((cur_lr * m_hat / (v_hat.sqrt() + args.eps)).to(p.dtype))

        if is_eval_round:
            tl, ta = evaluate(model, x_test, y_test, device)
            el = time.time() - t0
            sig_med = float(np.median(sig_window)) if sig_window else None
            if sig_med is not None:
                probe["zo_signal_med"] = float(f"{sig_med:.6e}")
                probe["zo_signal_zero_frac"] = round(
                    float(np.mean([s == 0.0 for s in sig_window])), 4)
            sig_window = []
            logger.log(rnd, np.mean(running), tl, ta, el, lr=cur_lr, **probe)
            print(f"round {rnd:6d} | train_avg {np.mean(running):.4f} | "
                  f"test_loss {tl:.4f} | test_acc {ta:.4f} | {el:.0f}s")
            running = []
            if watch.update(rnd, sig_med, tl):
                print(f"[abort] round {rnd}: {watch.reason}")
                break
    logger.finalize(tl, ta, time.time() - t0, aborted=watch.reason)
    logger.record["rounds_done"] = rnd
    logger._dump()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def default_out(args):
    """조합을 파일명으로 인코딩 (aggregate 시 파일명만으로 조건 식별 가능하게).
    zo_weight_decay는 zo 모드, weight_decay는 fo 모드에서만 붙임."""
    wd = args.zo_weight_decay if args.mode.startswith("zo") else args.weight_decay
    tag = (f"{args.mode}_{args.model}_{args.dtype}_lr{args.lr:g}_mu{args.mu:g}"
           f"_b1{args.beta1:g}_b2{args.beta2:g}_wd{wd:g}"
           f"_bs{args.batch_size}_n{args.n_nodes}_r{args.n_rounds}")
    if args.warmup_rounds > 0:
        tag += f"_wu{args.warmup_rounds}"
    if args.lr_schedule != "constant":
        tag += f"_{args.lr_schedule}"
    if args.init != "default":
        tag += f"_{args.init}"
    if args.init_gain != 1.0:
        tag += f"_g{args.init_gain:g}"
    if args.fc_scale != 1.0:
        tag += f"_fc{args.fc_scale:g}"
    return os.path.join("results", tag + ".json")


def load_yaml_defaults(path, parser):
    """yaml의 키를 parser default로 주입. dest 이름과 일치해야 하며,
    CLI에서 명시한 인자가 항상 yaml을 이긴다 (argparse default 우선순위)."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: 최상위가 dict여야 함 (got {type(cfg).__name__})")
    valid = {a.dest for a in parser._actions}
    unknown = set(cfg) - valid
    if unknown:
        raise ValueError(f"{path}: 알 수 없는 키 {sorted(unknown)}  (사용 가능: {sorted(valid - {'help'})})")
    parser.set_defaults(**cfg)
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="yaml 설정 파일. 여기 적힌 값이 default가 되고, "
                        "CLI로 준 인자가 그걸 override함 (grid sweep 시 "
                        "--config base.yaml --mu 2.5e-3 식으로 사용)")
    p.add_argument("--mode", default="fo", choices=["fo", "zo_sign", "zo_adam"])
    p.add_argument("--model", default="resnet20", choices=list(MODELS))
    p.add_argument("--norm", default="auto", choices=["auto", "batch", "group"],
                   help="auto: fo→batch, zo→group (BN running stats 오염 회피)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp64"],
                   help="모델/forward dtype. ZO에서 bf16은 L+−L− 차이가 rounding에 "
                        "묻힐 수 있음(측정 정밀도 축의 일부). optimizer state와 z "
                        "생성/업데이트 계산은 dtype 무관하게 fp32(fp64면 fp64) 유지")
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
    p.add_argument("--precision_probe", action="store_true",
                   help="eval마다 같은 θ/z/배치로 ΔL을 fp64 재측정해서 "
                        "zo_delta / zo_delta_fp64 / zo_delta_err / zo_curv_fp64를 "
                        "history에 기록 (측정 노이즈 바닥 추적, ZO 모드 전용)")
    p.add_argument("--bias3_probe", action="store_true",
                   help="--precision_probe와 함께: μ와 μ/2에서 ΔL을 재고 Richardson으로 "
                        "3차항 bias를 직접 추출 → zo_bias3 / zo_grad_term 기록. "
                        "|zo_bias3 / zo_delta_fp64|가 실제 bias 비중 (fp64 forward 2회 추가)")
    p.add_argument("--curvature_check", action="store_true",
                   help="eval마다 무섭동 L0도 측정해서 (L+−L0) vs (L0−L−) 대칭성 검증. "
                        "zo_dplus / zo_dminus / zo_curv(=μ²·zᵀHz 추정) 기록 "
                        "(SPSA 선형근사 위반 = mu bias 추적, ZO 모드 전용)")
    p.add_argument("--n_nodes", type=int, default=1)
    p.add_argument("--init", default="default",
                   choices=["default", "kaiming_normal", "kaiming_uniform",
                            "xavier_normal", "xavier_uniform", "orthogonal"],
                   help="Conv/Linear weight 초기화 스킴")
    p.add_argument("--init_gain", type=float, default=1.0,
                   help="초기화된 weight 전체에 곱하는 스케일 (‖θ‖ 조절)")
    p.add_argument("--fc_scale", type=float, default=1.0,
                   help="마지막 fc weight에만 추가로 곱하는 스케일 "
                        "(<1이면 초기 로짓이 작아짐)")
    p.add_argument("--zero_init_residual", action="store_true",
                   help="각 블록의 두 번째 norm weight를 0으로 → 블록이 항등함수로 시작")
    p.add_argument("--early_abort", action="store_true",
                   help="|g^T z| 신호가 초기 대비 붕괴하고 loss가 uniform(ln C) 근처에 "
                        "머물면 조기 종료 (grid sweep에서 죽은 조합 낭비 방지)")
    p.add_argument("--abort_min_round", type=int, default=1000,
                   help="이 라운드 이전에는 조기 종료 판정 안 함")
    p.add_argument("--abort_ratio", type=float, default=100.0,
                   help="신호가 초기 기준값의 1/ratio 아래로 떨어지면 붕괴로 간주")
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

    # --config를 먼저 읽어 default로 주입한 뒤 최종 파싱 (CLI가 yaml을 이김)
    pre, _ = p.parse_known_args()
    if pre.config:
        cfg = load_yaml_defaults(pre.config, p)
        print(f"[config] {pre.config}: {len(cfg)} keys loaded")
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
    if args.precision_probe and args.dtype == "fp64":
        print("[warn] --precision_probe는 fp64 런에서 무의미 (자기 자신과 비교) → 비활성")
        args.precision_probe = False
    if args.precision_probe and not is_zo:
        print("[warn] --precision_probe는 zo 모드 전용 → 비활성")
        args.precision_probe = False
    if args.bias3_probe and not args.precision_probe:
        print("[info] --bias3_probe는 --precision_probe를 필요로 함 → 함께 활성화")
        args.precision_probe = is_zo
    if args.curvature_check and not is_zo:
        print("[warn] --curvature_check는 zo 모드 전용 → 비활성")
        args.curvature_check = False
    args.use_augment = (args.mode == "fo") if args.augment == "auto" else (args.augment == "on")
    if args.out is None:
        args.out = default_out(args)

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage = device if (args.gpu_resident and device == "cuda") else "cpu"

    data = load_cifar100_ram(label_key=args.label_key, storage_device=storage)
    num_classes = 100 if args.label_key == "fine_label" else 20
    DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}
    model = MODELS[args.model](num_classes=num_classes, norm=args.norm)
    model = init_model(model, scheme=args.init, gain=args.init_gain,
                       fc_scale=args.fc_scale,
                       zero_init_residual=args.zero_init_residual)
    model = model.to(device=device, dtype=DTYPES[args.dtype])
    stats = param_stats(model)
    n_params = stats["n_params"]
    print(f"[model] {args.model} ({args.norm} norm, {args.dtype}, init={args.init}"
          f"{f'×{args.init_gain:g}' if args.init_gain != 1.0 else ''}): "
          f"{n_params / 1e6:.2f}M params, mode={args.mode}, device={device}")
    if is_zo:
        # ZO 감각 잡기용: perturbation 크기 ‖μz‖ ≈ μ√d 와 ‖θ‖ 비교
        pert = args.mu * stats["sqrt_d"]
        print(f"[zo] ||theta||={stats['theta_norm']:.2f}  mu*sqrt(d)={pert:.2f}  "
              f"ratio={pert / stats['theta_norm']:.3f}"
              + ("  <- perturbation이 theta 대비 큼 (선형근사 의심)"
                 if pert / stats["theta_norm"] > 0.05 else ""))
    print(f"[out] {args.out}")

    logger = RunLogger(args, n_params)
    logger.record["init_stats"] = {k: round(v, 4) for k, v in stats.items()}
    if args.mode == "fo":
        train_fo(model, data, device, args, logger)
    else:
        train_zo(model, data, device, args, logger)
