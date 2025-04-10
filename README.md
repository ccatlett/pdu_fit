# pdu_fit
Ultra-fast parameter estimation for Pdu in vitro data

## Description

1. Simulates ODE system using diffrax and JIT compilation
2. Performs small batches of multiseeded Adam (via Latin Hypercube) using optax under Optuna parameter tuning
    - Trials pruned if sufficiently many seeds are pruned (see below)
3. With optimal Adam hyperparameters, performs large multiseeded Adam optimization. This includes
    - Pruning of trials where 1,2-PD does not fully decay in the simulated timespan
    - Normalization of all parameters (0, 1)
    - Options for unweighted/weighted by experimental error
    - Options for weighting in time (by inverse step-size, logarithmic)
    - Options for relative/normalized/unormalized error
    - Options for MSE/loglikelihood for heteroscedastic Gaussian noise
        - LL adds a learnable stdev (alpha)
4. Hessian-based UQ

Parallelized according to:
- Each Optuna trial as in (2) is its own slurm job
- Within each job, we construct N LHS seeds and perform the N optimizations using JAX pmap
- With optimal parameters, we distributed seeds of the larger LHS (3) as smaller slurm jobs

In (2), we save per job:
- All optimized seeds and their cost
- All simulations from optimized seeds
- Number of pruned seeds

In (3) we save per seed:
- All optimized seeds and their cost
- All simulations from optimized seeds
- Number of pruned seeds

## File contents

1. pdu_rhs.py
2. simulate.py
3. run_single_adam.py
4. run_single_optuna.py
5. slurm_optuna_job.sh
6. main.py
7. data directory
8. requirements.txt
9. environment.yml
10. README.md
11. results/optuna_tuning/trial_X/results.csv (ex)
12. results/seed_X/results.csv (ex)

