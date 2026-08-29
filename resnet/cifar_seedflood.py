"""
CIFAR-100 (HuggingFace uoft-cs/cifar100) + CIFAR-style ResNet
FO baseline + shared-seed ZO (SPSA) + projected-gradient, sign / Adam 모드

- torchvision 서버 이슈 우회: datasets 라이브러리로 로드
- 전체 데이터를 uint8 텐서로 RAM(또는 GPU)에 상주, DataLoader 없이 인덱스 샘플링

[추정기 3종]
  fo         : 일반 backprop. 비교 기준.
  zo_*       : shared-seed SPSA. 라운드당 seed Q개 broadcast → 전 노드가 같은 z에서
               θ±μz 두 지점의 로컬 배치 loss 평가 → 서버가 노드 평균 L̄±로
               scalar=(L̄+−L̄−)/(2μ) → pseudo_grad = mean_q(scalar_q·z_q).
               forward 2Q회, backward 없음.
  proj_*     : backward 1회로 g를 구한 뒤 s = Zg를 한 번에 계산 → 서버가
               Z^T s/Q로 재구성. 통신은 zo와 동일하게 (seed, 스칼라 Q개)뿐이라
               model size 무관이지만, 노드가 backward를 돌 수 있다고 가정한다.
               μ가 없으므로 곡률/3차항 bias도, 차분의 정밀도 소실도 발생하지 않는다.

  세 모드는 같은 통신 예산에서 비교할 수 있어서:
    zo vs proj : μ bias가 성능을 얼마나 깎는지 분리
    proj vs fo : 랜덤 부분공간(Q차원) 사영 자체의 손실만 측정
  (Gaussian Z이므로 유한 Q에서는 Q=d여도 매 표본에서 g를 완전 복원하지는 않는다.)

- 라운드의 z는 (directions, params) flat 행렬로 보관해 내적/재구성을 한 번에 수행
- θ 복원은 스냅샷 copy (zo만; proj는 섭동 없음)
- optimizer state(m,v)는 파라미터가 bf16이어도 fp32 master 유지

메모리: 데이터 train ≈154MB, test ≈31MB. 추가로 flat Z가 shared는 Q*d,
per_node는 N*Q*d 원소를 차지한다 (fp32 ResNet20, Q=64면 약 71MB).

주의(BN vs ZO): BatchNorm은 ±μ 두 번의 forward마다 running stats를 갱신해서 ZO의
loss 차이 측정을 오염시킴 → --norm auto는 zo/proj 모드에서 GroupNorm을 선택함.
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
_NORMALIZATION_CACHE = {}


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
    # 매 배치마다 작은 CPU tensor를 GPU로 복사하지 않도록 device별로 재사용한다.
    key = (x_uint8.device.type, x_uint8.device.index)
    if key not in _NORMALIZATION_CACHE:
        _NORMALIZATION_CACHE[key] = (
            CIFAR100_MEAN.to(x_uint8.device).view(1, 3, 1, 1),
            CIFAR100_STD.to(x_uint8.device).view(1, 3, 1, 1),
        )
    mean, std = _NORMALIZATION_CACHE[key]
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


def make_act(kind: str):
    """활성함수 팩토리.

    ZO 관점 메모: ReLU는 0에서 미분 불가능(piecewise linear)이라 SPSA의 테일러 전개
    전제가 깨지는 지점이 존재한다. μ 크기의 perturbation이 많은 뉴런의 부호를 뒤집으면
    L±가 서로 다른 선형 조각(linear region) 위에서 평가되어, ΔL이 국소 gradient가 아닌
    조각 간 점프를 재게 된다. GELU/SiLU/Softplus 같은 매끄러운 활성함수는 이 문제가
    없어서 ZO에서 더 안정적일 수 있다 — dead ReLU로 인한 gradient 소실도 함께 회피.
    """
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "softplus": nn.Softplus(beta=5.0),  # beta↑ 일수록 relu에 가까움
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
        "tanh": nn.Tanh(),
    }[kind]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, norm="batch", act="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.n1 = make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.n2 = make_norm(norm, out_ch)
        self.act1 = make_act(act)
        self.act2 = make_act(act)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                make_norm(norm, out_ch),
            )

    def forward(self, x):
        out = self.act1(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks, widths, num_classes=100, norm="batch", act="relu"):
        super().__init__()
        self.in_ch = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.n1 = make_norm(norm, widths[0])
        self.act1 = make_act(act)
        layers = []
        for i, (nb, w) in enumerate(zip(num_blocks, widths)):
            stride = 1 if i == 0 else 2
            layers.append(self._make_layer(w, nb, stride, norm, act))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(widths[-1], num_classes)

    def _make_layer(self, out_ch, n_blocks, stride, norm, act):
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_ch, out_ch, s, norm, act))
            self.in_ch = out_ch
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.act1(self.n1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20(num_classes=100, norm="batch", act="relu"):
    """~0.28M params — ZO 스크리닝용으로 현실적인 크기"""
    return ResNet([3, 3, 3], [16, 32, 64], num_classes, norm, act)


def resnet18(num_classes=100, norm="batch", act="relu"):
    """~11.2M params — FO baseline / 나중 스케일업용"""
    return ResNet([2, 2, 2, 2], [64, 128, 256, 512], num_classes, norm, act)


MODELS = {"resnet20": resnet20, "resnet18": resnet18}


# ----------------------------------------------------------------------------
# Parameter initialization
# ----------------------------------------------------------------------------


def init_model(model, scheme="default", gain=1.0, fc_scale=1.0,
               zero_init_residual=False, act="relu"):
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
    # kaiming의 nonlinearity 인자: 지원되는 것만 매핑, 나머지는 relu gain 사용
    kaiming_nl = {"relu": "relu", "leaky_relu": "leaky_relu", "tanh": "tanh",
                  "selu": "selu"}.get(act, "relu")
    if scheme != "default":
        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                w = mod.weight
                if scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity=kaiming_nl)
                elif scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(w, mode="fan_out", nonlinearity=kaiming_nl)
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
    bf16에서 직접 randn을 뽑지 않아 방향 저장과 행렬 연산의 정밀도를 유지한다."""
    return torch.float64 if p.dtype == torch.float64 else torch.float32


class FlatDirections:
    """Q개의 random direction을 contiguous (Q, d) 행렬로 보관한다.

    행 단위 생성 시 parameter-shaped view를 순서대로 채워 기존 seed replay 규칙을
    유지한다. 내적과 gradient 재구성은 각각 Z@g, Z.T@s 한 번으로 처리한다.
    """

    def __init__(self, params, capacity):
        self.params = params
        self.numels = [p.numel() for p in params]
        self.dtype = zo_gen_dtype(params[0])
        self.data = torch.empty(
            (capacity, sum(self.numels)), device=params[0].device, dtype=self.dtype
        )

    def fill_(self, seeds, device):
        directions = self.data[:len(seeds)]
        for row, seed in zip(directions, seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            for view in self.views(row):
                view.normal_(generator=generator)
        return directions

    def views(self, flat):
        return [chunk.view_as(p) for chunk, p in zip(flat.split(self.numels), self.params)]


def draw_seeds(count):
    """query마다 scalar를 꺼내지 않고 seed들을 한 번에 생성한다."""
    return torch.randint(0, 2**31 - 1, (count,)).tolist()


@torch.no_grad()
def zo_perturb_(params, direction_views, scale):
    """flat direction의 parameter-shaped view로 in-place perturb한다."""
    zs = direction_views
    if all(z.dtype == p.dtype for z, p in zip(zs, params)):
        torch._foreach_add_(params, zs, alpha=scale)
    else:  # bf16 파라미터: fp32에서 스케일 후 캐스팅
        scaled = torch._foreach_mul(zs, scale)
        torch._foreach_add_(params, [z.to(p.dtype) for z, p in zip(scaled, params)])


@torch.no_grad()
def zo_loss(model, batches):
    """노드별 로컬 배치 loss의 평균 (SeedFlood의 서버측 평균에 해당).
    forward는 모델 dtype 그대로 두되(=dtype 실험의 대상), CE 집계는 fp32로.
    배치마다 .item()을 부르지 않고 GPU에서 누적 → 라운드당 동기화 횟수 감소."""
    total = None
    for xb, yb in batches:
        l = F.cross_entropy(model(xb).float(), yb)
        total = l if total is None else total + l
    return float(total) / len(batches)


@torch.no_grad()
def precision_probe(model, params, theta, batches, direction, mu, bias3=False):
    """같은 θ / 같은 z / 같은 배치에서 L0, L±μ (옵션: L±μ/2)를 fp64로 재측정.

    - 호출 시점 조건: θ가 스냅샷(theta)으로 복원된 상태
    - 학습 경로에서 저장한 동일한 flat z를 fp64로 승격 (같은 방향 보장)
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
        chunks = direction.split([p.numel() for p in params64])
        for p_, t_, z_ in zip(params64, theta64, chunks):
            p_.data.copy_(t_ + offset * z_.view_as(p_).double())

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


def compute_grad(model, batches, params):
    """batches 평균 loss의 gradient를 contiguous flat vector로 반환."""
    model.zero_grad(set_to_none=True)
    total = torch.zeros((), device=params[0].device)
    for xb, yb in batches:
        loss = F.cross_entropy(model(xb).float(), yb) / len(batches)
        loss.backward()
        total += loss.detach()
    g = torch.cat([p.grad.detach().to(zo_gen_dtype(p)).flatten() for p in params])
    model.zero_grad(set_to_none=True)
    return g, float(total)


def run_fo_warmup(model, data, device, args, logger):
    """ZO 시작 전 FO(SGD+momentum)로 지정 step만큼 사전 학습.

    목적: 균등분포 흡수 상태(loss ≈ ln C, gradient 소실)를 FO로 먼저 빠져나온 뒤
    ZO에 넘겨서, "ZO가 학습을 못 하는가" vs "초기 탈출만 못 하는가"를 분리한다.
    통신량 O(1) 가정을 깨므로 배포 방식이 아니라 진단용.

    ZO Adam의 m/v는 이월하지 않고 0에서 시작한다 (train_zo가 새로 만듦).
    """
    x_train, y_train, x_test, y_test = data
    dtype = next(model.parameters()).dtype
    opt = torch.optim.SGD(model.parameters(), lr=args.fo_warmup_lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    model.train()
    t0 = time.time()
    running = []
    for step in range(1, args.fo_warmup_steps + 1):
        xb, yb = get_batch(x_train, y_train, args.batch_size, device,
                           augment=args.use_augment, dtype=dtype)
        loss = F.cross_entropy(model(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % args.eval_every == 0 or step == args.fo_warmup_steps:
            tl, ta = evaluate(model, x_test, y_test, device)
            print(f"[fo-warmup] step {step:5d} | train_avg {np.mean(running):.4f} | "
                  f"test_loss {tl:.4f} | test_acc {ta:.4f} | {time.time() - t0:.0f}s")
            running = []
    tl, ta = evaluate(model, x_test, y_test, device)
    logger.record["fo_warmup"] = {
        "steps": args.fo_warmup_steps,
        "lr": args.fo_warmup_lr,
        "handoff_test_loss": round(float(tl), 6),
        "handoff_test_acc": round(float(ta), 6),
        "sec": round(time.time() - t0, 1),
    }
    logger._dump()
    print(f"[fo-warmup] handoff → ZO at test_loss {tl:.4f}, acc {ta:.4f}")
    return model


def train_zo(model, data, device, args, logger):
    x_train, y_train, x_test, y_test = data
    if args.fo_warmup_steps > 0:
        run_fo_warmup(model, data, device, args, logger)
    model.train()  # (norm=group이면 train/eval 동작 동일, dropout 없음)
    params = list(model.parameters())
    dtype = next(model.parameters()).dtype
    node_indices = build_node_indices(x_train.size(0), args.n_nodes, seed=args.data_seed)
    n_directions = args.n_queries * (args.n_nodes if args.seed_mode == "per_node" else 1)
    direction_store = FlatDirections(params, n_directions)
    direction_mb = direction_store.data.numel() * direction_store.data.element_size() / 2**20
    print(f"[directions] shape={tuple(direction_store.data.shape)}, {direction_mb:.1f} MiB")

    # optimizer state는 파라미터가 bf16이어도 fp32 master로 유지
    # (bf16은 mantissa 8bit라 beta1=0.999 같은 계수 자체가 표현이 안 되고,
    #  (1-beta2)=1e-4 스케일 누적이 바로 소실됨)
    if not args.mode.endswith("_sign"):
        m = [torch.zeros(p.shape, device=device, dtype=zo_gen_dtype(p)) for p in params]
        v = [torch.zeros(p.shape, device=device, dtype=zo_gen_dtype(p)) for p in params]

    running, t0 = [], time.time()
    tl = ta = float("nan")
    watch = CollapseWatch(args, model.fc.out_features)
    sig_window = []  # 매 라운드 |g^T z| (probe 없어도 delta로 계산 가능)
    for rnd in range(1, args.n_rounds + 1):
        # 각 노드가 로컬 배치 하나씩 샘플.
        batches = [
            get_batch(x_train, y_train, args.batch_size, device,
                      indices=ni, augment=args.use_augment, dtype=dtype)
            for ni in node_indices
        ]

        # θ 스냅샷 복원 방식: 산술 복원(+μ→−2μ→+μ)은 bf16에서 rounding 잔차가
        # μ와 같은 스케일(~1e-2)로 남아 매 라운드 θ를 오염시킴 → copy로 정확 복원.
        # proj 모드는 파라미터를 섭동하지 않으므로 스냅샷이 필요 없다.
        theta = None if args.mode.startswith("proj") else \
            [p.detach().clone() for p in params]
        pdata = [p.data for p in params]  # leaf 제약 없이 in-place 복원용 뷰

        seeds = draw_seeds(n_directions)
        directions = direction_store.fill_(seeds, device)

        if args.mode.startswith("proj"):
            # ── projected gradient: backward 1회 + batched Z @ g ─────────────
            # 통신은 SPSA와 동일 (seed broadcast + 스칼라 Q개 업로드 → model size 무관)
            # 이지만 노드가 backward를 돌 수 있다고 가정한다. μ가 없어 bias/정밀도
            # 문제가 사라지므로, SPSA와 같은 통신 예산에서 비교하면 μ bias가 성능을
            # 얼마나 깎는지 분리해서 볼 수 있다.
            if args.seed_mode == "shared":
                g_true, loss_val = compute_grad(model, batches, params)
                scalars_tensor = torch.mv(directions, g_true)
            else:  # per_node: 노드마다 자기 gradient를 자기 seed에 사영
                scalar_parts, loss_val = [], 0.0
                g_true = None
                for node_index, nb in enumerate(batches):
                    g_i, l_i = compute_grad(model, [nb], params)
                    start = node_index * args.n_queries
                    node_directions = directions[start:start + args.n_queries]
                    scalar_parts.append(torch.mv(node_directions, g_i))
                    loss_val += l_i / len(batches)
                scalars_tensor = torch.cat(scalar_parts)
            scalars = scalars_tensor.tolist()  # 모든 query를 한 번의 device sync로 로깅
            l_plus = l_minus = loss_val  # 로그용 (차분이 없으므로 loss 그 자체)
        elif args.seed_mode == "shared":
            # 라운드당 Q개 seed를 전 노드에 broadcast. 모든 노드가 같은 z에서
            # 자기 배치 loss를 평가 → 서버는 노드 평균 L̄±로 scalar 하나를 만든다.
            # Q개 query 전부 같은 배치를 씀 → 배치 노이즈는 공통, z 방향 분산만 1/Q.
            scalars = []
            for query_index, direction in enumerate(directions):
                views = direction_store.views(direction)
                zo_perturb_(params, views, +args.mu)
                lp = zo_loss(model, batches)
                torch._foreach_copy_(pdata, theta)
                zo_perturb_(params, views, -args.mu)
                lm = zo_loss(model, batches)
                torch._foreach_copy_(pdata, theta)
                scalars.append((lp - lm) / (2 * args.mu))
                if query_index == 0:
                    l_plus, l_minus = lp, lm  # probe/로그는 첫 query 기준
        else:  # per_node
            # 노드마다 다른 seed. 각 노드가 자기 z로 자기 배치만 평가 → scalar_i.
            # 서버는 pseudo_grad를 평균: mean_i(scalar_i·z_i).
            # 노드마다 다른 random projection을 적용하며, 노드당 Q개씩 뽑아 총 N·Q
            # 방향을 쓴다 (shared의 Q개와 통신/계산 예산이 다름에 주의).
            scalars = []
            direction_index = 0
            for ni_batch in batches:
                for _ in range(args.n_queries):
                    views = direction_store.views(directions[direction_index])
                    zo_perturb_(params, views, +args.mu)
                    lp = zo_loss(model, [ni_batch])
                    torch._foreach_copy_(pdata, theta)
                    zo_perturb_(params, views, -args.mu)
                    lm = zo_loss(model, [ni_batch])
                    torch._foreach_copy_(pdata, theta)
                    scalars.append((lp - lm) / (2 * args.mu))
                    if direction_index == 0:
                        l_plus, l_minus = lp, lm
                    direction_index += 1
        if args.mode.startswith("zo"):
            scalars_tensor = torch.tensor(scalars, device=device,
                                          dtype=direction_store.dtype)

        scalar = float(np.mean(scalars))  # 진단/로그용 대표값
        running.append((l_plus + l_minus) / 2)
        sig_window.append(abs(scalar))  # |g^T z| 추정 (bias 미보정)
        cur_lr = lr_at(rnd, args)

        is_eval_round = (rnd % args.eval_every == 0 or rnd == args.n_rounds)
        probe = {}
        # l_plus/l_minus는 첫 seed 기준이므로 진단도 같은 배치 범위를 써야 함
        probe_batches = batches if args.seed_mode == "shared" else [batches[0]]
        if len(scalars) > 1 and is_eval_round:
            # query/노드 간 scalar 산포: 방향 추정 분산이 실제로 얼마나 되는지
            probe["zo_scalar_std"] = float(f"{float(np.std(scalars)):.6e}")
            probe["zo_scalar_mean"] = float(f"{scalar:.6e}")
            probe["zo_n_directions"] = len(scalars)
        if args.curvature_check and is_eval_round:
            # 대칭성 검증: (L+ − L0) =? (L0 − L−). 차이 = L+ + L− − 2L0 ≈ μ²·zᵀHz
            # θ는 지금 복원된 상태이므로 무섭동 loss를 같은 배치로 한 번 더 측정
            l_zero = zo_loss(model, probe_batches)
            dplus, dminus = l_plus - l_zero, l_zero - l_minus
            probe.update({
                "zo_dplus": float(f"{dplus:.6e}"),
                "zo_dminus": float(f"{dminus:.6e}"),
                "zo_curv": float(f"{dplus - dminus:.6e}"),
            })
        if args.precision_probe and is_eval_round:
            # θ 복원된 지금 시점에 같은 seed/배치로 fp64 재측정 (첫 seed 기준)
            delta_run = l_plus - l_minus
            pr = precision_probe(model, params, theta, probe_batches,
                                 directions[0], args.mu, bias3=args.bias3_probe)
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

        # 한 번의 matrix-vector product로 pseudo_grad = Z.T @ scalars / Q.
        want_cos = (args.mode.startswith("proj") and is_eval_round
                    and args.seed_mode == "shared")
        with torch.no_grad():
            if args.zo_weight_decay > 0:
                torch._foreach_mul_(params, 1 - cur_lr * args.zo_weight_decay)

            pseudo_flat = torch.mv(directions.t(), scalars_tensor).div_(len(seeds))
            pseudo_grad = direction_store.views(pseudo_flat)

            if want_cos:  # 재구성된 방향이 진짜 gradient와 얼마나 정렬됐나
                cos_dot = float(torch.dot(pseudo_flat, g_true).double())
                cos_pn = float(torch.linalg.vector_norm(pseudo_flat).double())
                gn = float(torch.linalg.vector_norm(g_true).double())
                denom = cos_pn * gn
                probe["proj_cosine"] = round(cos_dot / denom, 6) if denom > 0 else None
                probe["proj_grad_norm"] = float(f"{gn:.6e}")

            if args.mode.endswith("_sign"):
                steps = torch._foreach_sign(pseudo_grad)
                torch._foreach_mul_(steps, cur_lr)
            else:  # Adam
                torch._foreach_mul_(m, args.beta1)
                torch._foreach_add_(m, pseudo_grad, alpha=1 - args.beta1)
                torch._foreach_mul_(v, args.beta2)
                torch._foreach_addcmul_(v, pseudo_grad, pseudo_grad,
                                        value=1 - args.beta2)
                bc1 = 1 - args.beta1 ** rnd
                bc2 = 1 - args.beta2 ** rnd
                denom_t = torch._foreach_sqrt(v)
                torch._foreach_div_(denom_t, math.sqrt(bc2))
                torch._foreach_add_(denom_t, args.eps)
                steps = torch._foreach_div(m, denom_t)
                torch._foreach_mul_(steps, cur_lr / bc1)

            if steps[0].dtype == params[0].dtype:
                torch._foreach_sub_(params, steps)
            else:  # bf16 파라미터
                torch._foreach_sub_(params, [s.to(p.dtype)
                                             for s, p in zip(steps, params)])

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
    wd = args.zo_weight_decay if args.mode.startswith(("zo", "proj")) else args.weight_decay
    mu_tag = "" if args.mode.startswith(("fo", "proj")) else f"_mu{args.mu:g}"
    tag = (f"{args.mode}_{args.model}_{args.dtype}_lr{args.lr:g}{mu_tag}"
           f"_b1{args.beta1:g}_b2{args.beta2:g}_wd{wd:g}"
           f"_bs{args.batch_size}_n{args.n_nodes}_r{args.n_rounds}")
    if args.n_queries > 1:
        tag += f"_q{args.n_queries}"
    if args.seed_mode != "shared":
        tag += f"_{args.seed_mode}"
    if args.warmup_rounds > 0:
        tag += f"_wu{args.warmup_rounds}"
    if args.lr_schedule != "constant":
        tag += f"_{args.lr_schedule}"
    if args.init != "default":
        tag += f"_{args.init}"
    if args.act != "relu":
        tag += f"_{args.act}"
    if args.init_gain != 1.0:
        tag += f"_g{args.init_gain:g}"
    if args.fc_scale != 1.0:
        tag += f"_fc{args.fc_scale:g}"
    if args.fo_warmup_steps > 0:
        tag += f"_fowu{args.fo_warmup_steps}@{args.fo_warmup_lr:g}"
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
    p.add_argument("--mode", default="fo",
                   choices=["fo", "zo_sign", "zo_adam", "proj_adam", "proj_sign"],
                   help="fo: 일반 backprop. zo_*: SPSA (forward 2Q회, μ 필요). "
                        "proj_*: backward 1회로 g를 구한 뒤 s=Zg, g_hat=Z^T s/Q를 "
                        "행렬 연산으로 계산. 통신량은 zo와 동일하지만 "
                        "μ가 없어 곡률/3차항 bias와 차분 정밀도 소실이 사라짐")
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
    p.add_argument("--seed_mode", default="shared", choices=["shared", "per_node"],
                   help="shared: 라운드당 seed를 전 노드에 broadcast, 모든 노드가 같은 z에서 "
                        "자기 배치를 평가 → 서버가 L̄±로 scalar 하나 (SeedFlood 기본). "
                        "per_node: 노드마다 다른 seed, 각자 자기 배치만 평가한 뒤 "
                        "pseudo_grad를 평균 → cross term(g_A^T z_B)이 노이즈로 남음. "
                        "방향 수가 shared는 Q개, per_node는 N·Q개라 예산이 다름에 주의")
    p.add_argument("--n_queries", type=int, default=1,
                   help="라운드당 z 방향 개수 Q (multi-query SPSA). "
                        "pseudo_grad = mean_q(scalar_q·z_q)로 방향 추정 분산이 1/Q. "
                        "forward 비용은 2Q배, 통신량은 (seed,scalar)×Q로 여전히 "
                        "model size 무관. 모든 query가 같은 배치를 쓰므로 배치 노이즈는 "
                        "안 줄고 z 방향 분산만 줄어듦 (batch_size 축과 분리)")
    p.add_argument("--n_nodes", type=int, default=1)
    p.add_argument("--act", default="relu",
                   choices=["relu", "gelu", "silu", "softplus", "elu",
                            "leaky_relu", "tanh"],
                   help="활성함수. ReLU는 0에서 미분 불가라 μ 섭동이 뉴런 부호를 뒤집으면 "
                        "L±가 서로 다른 linear region에서 평가됨 → gelu/silu/softplus 같은 "
                        "매끄러운 함수가 ZO에서 더 안정적일 수 있음")
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
    p.add_argument("--fo_warmup_steps", type=int, default=0,
                   help="ZO 시작 전 FO(SGD+momentum)로 학습할 step 수. "
                        "흡수 상태 탈출 후 ZO로 넘겨서 'ZO가 못 배우는가 vs 탈출만 "
                        "못 하는가'를 분리하는 진단용 (O(1) 통신 가정을 깨므로 배포용 아님). "
                        "ZO Adam의 m/v는 이월하지 않고 0에서 시작")
    p.add_argument("--fo_warmup_lr", type=float, default=0.05,
                   help="FO warmup 구간의 SGD lr (ZO의 --lr과 별개)")
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

    is_zo = args.mode.startswith("zo")       # SPSA (μ 차분)
    is_proj = args.mode.startswith("proj")   # backward + 랜덤 사영
    is_dist = is_zo or is_proj               # 분산 추정기 계열 (train_zo가 처리)
    if args.lr is None:
        args.lr = 0.1 if args.mode == "fo" else 1e-3
    if args.norm == "auto":
        # proj도 group으로: zo와 같은 아키텍처여야 공정 비교가 됨
        args.norm = "group" if is_dist else "batch"
    if args.lr_schedule == "auto":
        args.lr_schedule = "constant" if is_dist else "cosine"
    if is_zo and args.norm == "batch":
        print("[warn] zo + BatchNorm: ±μ forward마다 running stats가 갱신되어 "
              "측정이 오염될 수 있음 (--norm group 권장)")
    if args.precision_probe and args.dtype == "fp64":
        print("[warn] --precision_probe는 fp64 런에서 무의미 (자기 자신과 비교) → 비활성")
        args.precision_probe = False
    if args.precision_probe and not is_zo:
        print("[warn] --precision_probe는 zo(SPSA) 모드 전용 → 비활성")
        args.precision_probe = False
    if args.fo_warmup_steps > 0 and not is_dist:
        print("[warn] --fo_warmup_steps는 zo/proj 모드 전용 → 무시")
        args.fo_warmup_steps = 0
    if args.bias3_probe and not args.precision_probe:
        print("[info] --bias3_probe는 --precision_probe를 필요로 함 → 함께 활성화")
        args.precision_probe = is_zo
    if args.curvature_check and not is_zo:
        print("[warn] --curvature_check는 zo(SPSA) 모드 전용 → 비활성")
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
    model = MODELS[args.model](num_classes=num_classes, norm=args.norm, act=args.act)
    model = init_model(model, scheme=args.init, gain=args.init_gain,
                       fc_scale=args.fc_scale,
                       zero_init_residual=args.zero_init_residual, act=args.act)
    model = model.to(device=device, dtype=DTYPES[args.dtype])
    stats = param_stats(model)
    n_params = stats["n_params"]
    print(f"[model] {args.model} ({args.norm} norm, {args.act}, {args.dtype}, "
          f"init={args.init}"
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
