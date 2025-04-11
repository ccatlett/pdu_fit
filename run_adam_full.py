import jax
import jax.numpy as jnp
import optax as opx
from run_adam_tune import adam_step

def run_adam_with_outputs(params_init, adam_lr, adam_b1, adam_b2, num_steps, objective_fn):
    """
    Run the Adam optimizer on the given parameters and return the final parameters, loss, and grad.
    
    Args:
        params_init (array): initial parameters for the optimization
        adam_lr (float): learning rate for Adam optimizer
        adam_b1 (float): beta1 parameter for Adam optimizer
        adam_b2 (float): beta2 parameter for Adam optimizer
        num_steps (int): number of optimization steps
        objective_fn (function): objective function to minimize

    Returns:
        dict: dictionary containing the final parameters, loss, grad, and convergence status
    """

    optimizer = opx.adam(learning_rate=adam_lr, b1=adam_b1, b2=adam_b2)
    opt_state = optimizer.init(params_init)

    def step_fn(carry, _):
        params, opt_state = carry
        params, opt_state, loss = adam_step(params, opt_state, optimizer, objective_fn)
        return (params, opt_state), loss

    (final_params, _), losses = jax.lax.scan(step_fn, (params_init, opt_state), None, length=num_steps)

    final_loss = losses[-1]
    loss_diff = jnp.abs(losses[-1] - losses[-2])
    converged = loss_diff < 1e-3

    return {
        "optimal_params": final_params,
        "loss": final_loss,
        "converged": bool(converged)
    }
