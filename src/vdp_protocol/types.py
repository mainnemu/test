from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PipelineOutput:
    R_slope: np.ndarray      # (12, 2, 10)  [Config, Bpol, Rep]
    resid: np.ndarray        # (12, 5, 2, 10) [Config, Step, Bpol, Rep]
    R_bar: np.ndarray        # (12, 2)      [Config, Bpol] (Mean over reps)
    R_even: np.ndarray       # (12,)
    R_odd: np.ndarray        # (12,)
    sigma_p: float           # Pooled standard deviation
