import optuna
import os
import warnings
import jax.numpy as jnp
from run_adam_tune import batch_run_adam

warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

def tune_adam_hyperparams(objective_fn, seeds, num_steps=500, num_trials=20):
    def optuna_objective(trial):
        adam_lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        adam_b1 = trial.suggest_uniform('b1', 0.8, 0.99)
        adam_b2 = trial.suggest_uniform('b2', 0.9, 0.999)
        losses = batch_run_adam(seeds, adam_lr, adam_b1, adam_b2, num_steps, objective_fn)
        return float(jnp.min(losses))

    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, num_trials, n_jobs=os.cpu_count())
    return study.best_params
