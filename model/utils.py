import torch
import numpy as np
import random
import warnings
import torch.optim as opt

def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e

def matrix_inverse(m):
    m_inv = torch.cholesky_solve(torch.eye(m.shape[0], dtype=m.dtype, device=m.device), psd_safe_cholesky(m))
    return m_inv

def set_optimizer(optimizer, basemodel, weight, theta, lr_basemodel, lr_weight, lr_theta, use_basemodel):

    if use_basemodel:
        param = [{"params":filter(lambda p: p.requires_grad, basemodel.parameters())},\
                {"params": theta, "lr":lr_theta},\
                {"params": weight, "lr":lr_weight}]
        lr = lr_basemodel
    else:
        param = [{"params": theta},\
                {"params": weight, "lr":lr_weight}]
        lr = lr_theta       
    if optimizer == 'SGD':
        return opt.SGD(param, lr=lr)
    elif optimizer == 'Adam':
        return opt.Adam(param, lr=lr, weight_decay=1e-2)
    elif optimizer == 'AdamW':
        return opt.AdamW(param, lr=lr, weight_decay=1e-2)
    else: 
        raise TypeError("Only support SGD, Adam and AdamW optimizer")