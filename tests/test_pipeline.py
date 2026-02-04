import numpy as np
import pytest
from vdp_protocol.constants import S, N_REP, I_A
from vdp_protocol.pipeline import run_pipeline
from vdp_protocol.gates import gate0_linearity, gate1_reciprocity

def create_synthetic_data(R_true=100.0, add_noise=False):
    # Shape: (12, 5, 2, 10, 2)
    # V = R * I
    data = np.zeros((12, S, 2, N_REP, 2), dtype=np.float64)

    for s in range(S):
        i_val = I_A[s]
        # V(+I)
        data[:, s, :, :, 0] = R_true * i_val
        # V(-I)
        data[:, s, :, :, 1] = R_true * (-i_val)

    if add_noise:
        # Add tiny noise to create non-zero sigma_p
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-6, size=data.shape)
        data += noise

    return data

def test_pipeline_perfect_linear():
    # Case: Perfectly linear, Reciprocity holds, No noise
    data = create_synthetic_data(R_true=123.45, add_noise=False)
    pout = run_pipeline(data)

    # 1. R_hat should match R_true
    assert np.allclose(pout.R_slope, 123.45)
    # 2. Residuals should be 0
    assert np.allclose(pout.resid, 0.0)
    # 3. R_even should match R_true
    assert np.allclose(pout.R_even, 123.45)
    # 4. R_odd should be 0 (since +B and -B are identical)
    assert np.allclose(pout.R_odd, 0.0)
    # 5. Sigma_p should be 0
    assert pout.sigma_p == 0.0

    # Gate 0 Check
    t0 = gate0_linearity(pout)
    assert t0 == 0.0

def test_pipeline_with_noise():
    # Case: Small noise to verify Gate 1 logic and sigma_p calculation
    data = create_synthetic_data(R_true=100.0, add_noise=True)
    pout = run_pipeline(data)

    assert pout.sigma_p > 0.0

    # Since base data is identical for all configs, R_even should be roughly equal
    # but noise makes them slightly different.
    # Reciprocity should be very good (T1 small).
    t1 = gate1_reciprocity(pout)
    assert t1 < 5.0 # Should be small (normalized by sigma)
