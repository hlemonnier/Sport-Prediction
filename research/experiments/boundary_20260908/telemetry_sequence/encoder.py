"""Small causal CPC representation; no lap labels, raw I/O or sample selection.

The runner must freeze inputs and its same-driver/race, 180-second-exclusion
sampler before calling fit_cpc. Future raw tokens are training targets only;
embeddings() accepts context tokens alone. Both trained controls start from the
same initialization and must receive the same externally frozen batch schedule.

CPC objective: https://arxiv.org/abs/1807.03748 (application-specific adaptation).
Suggested commit: research(f1-live): implement controlled causal telemetry CPC
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20260908
INPUT_DIM = 34
MAX_TOKENS = 128
WIDTH = 24
EMBEDDING_DIM = 16
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
HORIZONS = (4, 16, 32)
NEGATIVES = 31
TEMPERATURE = .1
TRAIN_STEPS = 800
BATCH_SIZE = 32
LEARNING_RATE = .001
GRADIENT_LIMIT = 1.
CONTROLS = ('ordered', 'permuted', 'random')


def _tokens(values, *, ndim):
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.device.type != 'cpu':
        raise TypeError('Tokens must be float32 CPU tensors')
    if values.ndim != ndim or values.shape[-1] != INPUT_DIM or any(n == 0 for n in values.shape):
        raise ValueError('Invalid token tensor shape')
    if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 5):
        raise ValueError('Token measurements must be finite and scaled to [0,5]')
    padding = values[..., -1]
    if torch.any((padding != 0) & (padding != 1)):
        raise ValueError('Padding indicator must be binary')
    if torch.any(values[..., :-1][padding == 1] != 0):
        raise ValueError('Padding tokens must contain only their padding indicator')
    return values


def _contexts(values):
    _tokens(values, ndim=3)
    if not 1 <= values.shape[1] <= MAX_TOKENS:
        raise ValueError('Contexts contain at most128 tokens')
    # Padding is a prefix. A real packet followed by padding is not admissible.
    if torch.any(values[:, 1:, -1] > values[:, :-1, -1]):
        raise ValueError('Only left padding is permitted')
    return values


class CausalLayer(nn.Module):
    def __init__(self, inputs, dilation):
        super().__init__()
        self.left_padding = 2*dilation
        self.conv = nn.Conv1d(inputs, WIDTH, kernel_size=3, dilation=dilation, padding=0)
        # Input to LayerNorm is B,T,C: statistics never span time or batch.
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, values):
        values = self.conv(F.pad(values.transpose(1, 2), (self.left_padding, 0))).transpose(1, 2)
        return F.gelu(self.norm(values))


class CausalCPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(CausalLayer(INPUT_DIM if i == 0 else WIDTH, dilation)
                                   for i, dilation in enumerate(DILATIONS))
        self.projection = nn.Linear(WIDTH, EMBEDDING_DIM)
        self.target_encoder = nn.Linear(INPUT_DIM, EMBEDDING_DIM)
        self.heads = nn.Parameter(torch.empty(len(HORIZONS), EMBEDDING_DIM, EMBEDDING_DIM))
        for head in self.heads:
            nn.init.xavier_uniform_(head)

    def forward(self, contexts):
        """All causal timestamp states, B,T,16; useful for prefix regressions."""
        values = _contexts(contexts)
        for layer in self.layers:
            values = layer(values)
        return self.projection(values)

    def representation(self, contexts):
        """Newest-token unit vector; no pooling and no future-token inputs."""
        values = F.normalize(self(contexts)[:, -1, :], p=2, dim=-1, eps=1e-12)
        # An empty context has an explicit zero representation. The lap model's
        # unsupported fallback remains a separate, exact incumbent decision.
        return torch.where((contexts[:, -1, -1] == 1).unsqueeze(-1), torch.zeros_like(values), values)


def new_model(*, frozen=False):
    """Identical fixed initialization without advancing the caller's RNG."""
    if type(frozen) is not bool:
        raise TypeError('frozen must be boolean')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = CausalCPC().cpu().float()
    if parameter_count(model) >= 25_000:
        raise ValueError('CPC parameter budget exceeded')
    if frozen:
        model.requires_grad_(False)
        model.eval()
    return model


def parameter_count(model):
    return sum(value.numel() for value in model.parameters())


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode());digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CPCBatch:
    contexts: torch.Tensor  # B,128,34
    positives: torch.Tensor  # B,3,34: packet horizons4,16,32 in that order
    negatives: torch.Tensor  # B,31,34, common negative pool across the horizons
    context_keys: tuple[str, ...]  # metadata-only event/driver/endpoint identities


def validate_batch(batch, *, training=False):
    if not isinstance(batch, CPCBatch):
        raise TypeError('CPCBatch is required')
    contexts = _contexts(batch.contexts)
    _tokens(batch.positives, ndim=3);_tokens(batch.negatives, ndim=3)
    size = contexts.shape[0]
    if contexts.shape[1] != MAX_TOKENS or (training and size != BATCH_SIZE):
        raise ValueError('CPC uses128 context tokens and training batch32')
    if batch.positives.shape != (size, len(HORIZONS), INPUT_DIM) or batch.negatives.shape != (size, NEGATIVES, INPUT_DIM):
        raise ValueError('CPC requires3 positives and31 shared negatives per context')
    if torch.any(contexts[:, -1, -1] != 0) or torch.any(batch.positives[..., -1] != 0) or torch.any(batch.negatives[..., -1] != 0):
        raise ValueError('CPC endpoints and all positive/negative tokens must be real')
    if (not isinstance(batch.context_keys, tuple) or len(batch.context_keys) != size
            or any(not isinstance(key, str) or not key for key in batch.context_keys)):
        raise ValueError('Each context requires its nonempty metadata key')
    return batch


def transformed_contexts(contexts, context_keys, *, control):
    if control not in CONTROLS:
        raise ValueError('Unknown representation control')
    _contexts(contexts)
    if (not isinstance(context_keys, tuple) or len(context_keys) != len(contexts)
            or any(not isinstance(key, str) or not key for key in context_keys)):
        raise ValueError('Each context requires its nonempty metadata key')
    if control != 'permuted':
        return contexts
    # One implementation for training and inference, owned by the token module.
    from .tokens import permute_context
    rows = [permute_context(row.detach().cpu().numpy(), key=key, seed=SEED)
            for row, key in zip(contexts, context_keys, strict=True)]
    return torch.tensor(np.stack(rows), dtype=torch.float32, device='cpu')


def cpc_logits(model, batch, *, control='ordered'):
    validate_batch(batch)
    if control == 'random':
        raise ValueError('The random representation is never pretrained')
    contexts = transformed_contexts(batch.contexts, batch.context_keys, control=control)
    c = model.representation(contexts)
    positive = F.normalize(model.target_encoder(batch.positives), p=2, dim=-1, eps=1e-12)
    negative = F.normalize(model.target_encoder(batch.negatives), p=2, dim=-1, eps=1e-12)
    projected = torch.einsum('bd,kde->bke', c, model.heads)
    positive_logits = (projected*positive).sum(-1, keepdim=True)
    negative_logits = torch.einsum('bkd,bnd->bkn', projected, negative)
    return torch.cat((positive_logits, negative_logits), dim=-1)/TEMPERATURE


def cpc_loss(model, batch, *, control='ordered'):
    logits = cpc_logits(model, batch, control=control)
    labels = torch.zeros(logits.shape[0]*len(HORIZONS), dtype=torch.long, device='cpu')
    return F.cross_entropy(logits.reshape(-1, NEGATIVES+1), labels, reduction='mean')


def new_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False, fused=False)


def train_step(model, optimizer, batch, *, control='ordered'):
    """One bounded step, also used for synthetic gradient/reproducibility tests."""
    model.train();optimizer.zero_grad(set_to_none=True)
    loss = cpc_loss(model, batch, control=control)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite CPC objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_LIMIT, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError('Nonfinite updated CPC parameters')
    return {'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm.detach())}


def fit_cpc(batch_for_step: Callable[[int], CPCBatch], *, control, deadline_monotonic: float):
    """Exactly800 CPU steps. Both fits must share the runner's60-minute deadline.

    batch_for_step must replay the same frozen identities for both controls.
    This function performs no sampling and cannot establish source-year, event,
    driver or180-second exclusion rules; those are mandatory sampler contracts.
    """
    if control not in ('ordered', 'permuted'):
        raise ValueError('Exactly the ordered and permuted encoders are trained')
    if not callable(batch_for_step):
        raise TypeError('A frozen index-addressable batch schedule is required')
    if (isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic) or deadline_monotonic <= time.monotonic()):
        raise ValueError('An unexpired shared monotonic deadline is required')
    if torch.get_num_threads() != 1:
        raise ValueError('Set torch CPU intra-op threads to1 before fitting')
    model = new_model();initial = state_digest(model);optimizer = new_optimizer(model)
    started = time.monotonic();trace = []
    for step in range(TRAIN_STEPS):
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError('Shared pretraining deadline reached; no completed encoder')
        batch = validate_batch(batch_for_step(step), training=True)
        trace.append(train_step(model, optimizer, batch, control=control))
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError('Shared pretraining deadline reached after step; no completed encoder')
    model.eval();model.requires_grad_(False)
    return model, {'control': control, 'steps': TRAIN_STEPS, 'batch_size': BATCH_SIZE,
        'seed': SEED, 'learning_rate': LEARNING_RATE, 'gradient_limit': GRADIENT_LIMIT,
        'initial_state_sha256': initial, 'final_state_sha256': state_digest(model),
        'parameters': parameter_count(model), 'elapsed_seconds': time.monotonic()-started,
        'losses': [entry['loss'] for entry in trace],
        'gradient_norm_before_clip': [entry['gradient_norm_before_clip'] for entry in trace],
        'checkpoint_policy': 'final_fixed_step_only'}


def embeddings(model, contexts, *, context_keys, control='ordered'):
    """Inference-only16-vectors from admitted prefixes; never consumes targets."""
    if isinstance(contexts, np.ndarray):
        if contexts.dtype != np.float32:
            raise TypeError('Array tokens must be float32')
        contexts = torch.tensor(contexts, dtype=torch.float32, device='cpu')
    values = transformed_contexts(contexts, context_keys, control=control)
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model.representation(values).detach().cpu().numpy().copy()
    finally:
        model.train(previous)
    if result.shape != (len(values), EMBEDDING_DIM) or not np.isfinite(result).all():
        raise ValueError('Invalid causal representation')
    return result


def synthetic_benchmark(*, steps=10):
    """Small synthetic timing probe, never a predictive/historical experiment."""
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('Benchmark is limited to1..20 synthetic steps')
    if torch.get_num_threads() != 1:
        raise ValueError('Benchmark requires one CPU thread')
    rng = np.random.default_rng(SEED)
    def values(shape):
        array = rng.uniform(0, 1, shape).astype(np.float32);array[..., -1] = 0
        return torch.tensor(array)
    batch = CPCBatch(values((32,128,34)), values((32,3,34)), values((32,31,34)), tuple(f'synthetic/{i}' for i in range(32)))
    model=new_model();optimizer=new_optimizer(model)
    train_step(model, optimizer, batch)  # fixed warmup excluded from timing
    started=time.monotonic()
    for _ in range(steps): train_step(model, optimizer, batch)
    elapsed=time.monotonic()-started
    return {'synthetic_only': True, 'parameters': parameter_count(model), 'timed_steps': steps,
        'seconds_per_step': elapsed/steps, 'projected_two_800_step_seconds': 1600*elapsed/steps,
        'context_tensor_bytes': batch.contexts.nelement()*batch.contexts.element_size(),
        'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads(),
        'projection_limit':'Synthetic compute only; excludes historical token sampling, I/O and contention.'}
