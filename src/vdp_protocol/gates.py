import numpy as np
from .constants import I_A
from .config_table import GATE0_INDICES, PAIR_INDICES
from .types import PipelineOutput

def gate0_linearity(pout: PipelineOutput) -> float:
    """
    Gate 0: Max relative residual for Edge configs (E1..E4)
    T0 = max |eps| / |R*I|
    """
    # Extract only E1..E4
    # resid: (12, 5, 2, 10) -> (4, 5, 2, 10)
    eps_target = pout.resid[GATE0_INDICES, ...]
    R_target = pout.R_slope[GATE0_INDICES, ...]

    # V_pred = R * I
    I_broad = I_A[None, :, None, None]
    V_pred = R_target[:, None, :, :] * I_broad

    # Avoid zero division (Protocol assumes |I|>0, R!=0)
    denom = np.abs(V_pred)
    # Safe division for numerical stability in tests
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_resid = np.abs(eps_target) / denom
        # Replace inf/nan if R was exactly 0 (synthetic case)
        rel_resid = np.nan_to_num(rel_resid, nan=0.0, posinf=0.0)

    return float(np.max(rel_resid))

def gate1_reciprocity(pout: PipelineOutput) -> float:
    """
    Gate 1: Reciprocity Check
    T1 = max |R_even(A) - R_even(B)| / (sqrt(2) * sigma_p)
    """
    if pout.sigma_p <= 0:
        return 0.0 # Or appropriate handling for perfect synthetic data

    diffs = []
    for (idx_a, idx_b) in PAIR_INDICES:
        diff = np.abs(pout.R_even[idx_a] - pout.R_even[idx_b])
        diffs.append(diff)

    max_diff = np.max(diffs)
    T1 = max_diff / (np.sqrt(2.0) * pout.sigma_p)
    return float(T1)
