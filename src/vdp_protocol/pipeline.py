import numpy as np
from .constants import S, N_REP, I_A
from .types import PipelineOutput

def run_pipeline(data: np.ndarray) -> PipelineOutput:
    """
    Fixed Pipeline v2.2 (No branching)
    Input data: (12, S, 2, N_REP, 2) -> (C, S, B, R, Pol)
    """
    # Shape Assertions
    assert data.shape == (12, S, 2, N_REP, 2), f"Invalid shape: {data.shape}"

    # Step 1: Thermal EMF Removal (Current Reversal)
    # V_tilde = (V(+I) - V(-I)) / 2
    # Input index 4: 0=+I, 1=-I
    V_tilde = (data[..., 0] - data[..., 1]) / 2.0  # (12, 5, 2, 10)

    # Step 2: Ohmic Regression (Origin Constrained)
    # R = sum(I * V) / sum(I^2)
    # I_A shape (S,) -> Broadcast to (12, S, 2, 10)
    I_broad = I_A[None, :, None, None]

    num = np.sum(V_tilde * I_broad, axis=1)  # Sum over S -> (12, 2, 10)
    denom = np.sum(I_A**2)
    R_slope = num / denom                    # (12, 2, 10)

    # Calculate Residuals
    # V_pred = R * I
    V_pred = R_slope[:, None, :, :] * I_broad # (12, 5, 2, 10)
    resid = V_tilde - V_pred

    # Step 3: B-Field Separation
    # Mean over reps first
    R_bar = np.mean(R_slope, axis=2)         # (12, 2)

    # Even/Odd (Index 1: 0=+B, 1=-B)
    R_even = (R_bar[:, 0] + R_bar[:, 1]) / 2.0 # (12,)
    R_odd  = (R_bar[:, 0] - R_bar[:, 1]) / 2.0 # (12,)

    # Step 4: Variance Pooling
    # s^2 per config per B (ddof=1)
    # var over axis 2 (Reps) -> (12, 2)
    s2 = np.var(R_slope, axis=2, ddof=1)

    # Pool: Mean of variances (since N is constant)
    # sigma_p = sqrt( mean(s2) )
    sigma_p2 = np.mean(s2)
    sigma_p = np.sqrt(sigma_p2)

    return PipelineOutput(
        R_slope=R_slope,
        resid=resid,
        R_bar=R_bar,
        R_even=R_even,
        R_odd=R_odd,
        sigma_p=float(sigma_p)
    )
