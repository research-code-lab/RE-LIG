# Global RE-LIG hyperparameters — all XAI scripts import from this single source.
# Final configuration (see Section 4.2): n_steps=50, nt_samples=30, stdev=0.05.
# nt_samples=30 selected via grid search over {5,10,15,20,25,30}; performance
# gains exhibit diminishing returns beyond 30 (Section 3.4.2.A). stdev=0.05 sets
# the Gaussian perturbation magnitude; nt_type=smoothgrad.

RELIG_CONFIG = {
    'n_steps': 50,
    'nt_samples': 30,
    'stdev': 0.05,
    'nt_type': 'smoothgrad',
}
