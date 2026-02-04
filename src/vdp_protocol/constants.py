import numpy as np

# v2.2 Fixed Constants
S = 5
N_REP = 10
B_POLS = 2  # +B, -B
I_POLS = 2  # +I, -I

# Fixed Current Steps (Zero excluded)
# Shape: (S,)
I_mA = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
I_A = I_mA * 1e-3
