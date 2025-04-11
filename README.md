# pdu_fit
Ultra-fast parameter estimation for Pdu in vitro data

## Description

1. Simulates ODE system using diffrax and JIT compilation
2. Performs small batches of multiseeded Adam (via Latin Hypercube) using optax under Optuna parameter tuning
3. With optimal Adam hyperparameters, performs large multiseeded Adam optimization. This includes
    - Options for unweighted/weighted by experimental error
    - Options for weighting in time (by inverse step-size, logarithmic)
    - Options for relative/normalized/unormalized error
    - Options for MSE/L2
