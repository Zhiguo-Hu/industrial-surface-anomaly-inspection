"""Explicit local-runtime reproducibility settings and canonical tensor-state hashes."""
import hashlib
import os
import random
import numpy as np
import torch


def configure(seed=42, deterministic=False):
    if deterministic:
        if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    return {'seed':seed,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch':str(torch.__version__),'cuda_build':str(torch.version.cuda),
            'scope':'Same code, data, hardware/software runtime only; not a cross-version reproducibility guarantee'}


def state_hash(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()
