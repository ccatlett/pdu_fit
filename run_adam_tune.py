import jax
import optax as opx

###############################
#  Adam for single param set  #
###############################

def adam_step(params, opt_state, optimizer, objective_fn):
    """
    Perform a single step of the Adam optimizer.
    
    Args:
        params (array): parameters to optimize
        opt_state (object): optimizer state
        optimizer (object): optimizer object
        objective_fn (function): objective function to minimize
        
    Returns:
        tuple: updated parameters, optimizer state, and loss value
    """

    loss, grads = jax.value_and_grad(objective_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = opx.apply_updates(params, updates)
    return new_params, opt_state, loss

def run_adam_loop(params_init, adam_lr, adam_b1, adam_b2, num_steps, objective_fn):
    """
    Run the Adam optimization loop for a specified number of steps.
    
    Args:
        params_init (array): initial parameters
        adam_lr (float): learning rate for Adam optimizer
        adam_b1 (float): beta1 parameter for Adam optimizer
        adam_b2 (float): beta2 parameter for Adam optimizer
        num_steps (int): number of optimization steps
        objective_fn (function): objective function to minimize

    Returns:
        float: final loss value"""
    
    optimizer = opx.adam(learning_rate=adam_lr, b1=adam_b1, b2=adam_b2)
    opt_state = optimizer.init(params_init)

    def step_fn(carry, _):
        params, opt_state = carry
        params, opt_state, _ = adam_step(params, opt_state, optimizer, objective_fn)
        return (params, opt_state), None

    (final_params, _), _ = jax.lax.scan(step_fn, (params_init, opt_state), None, length=num_steps)
    final_loss = objective_fn(final_params)
    return final_loss

###############################
#  Loop over many param sets  #
###############################

def batch_run_adam(param_batch, adam_lr, adam_b1, adam_b2, num_steps, objective_fn):
    """
    Run the Adam optimization loop for a batch of parameters.
    
    Args:
        param_batch (array): batch of initial parameters
        adam_lr (float): learning rate for Adam optimizer
        adam_b1 (float): beta1 parameter for Adam optimizer
        adam_b2 (float): beta2 parameter for Adam optimizer
        num_steps (int): number of optimization steps
        objective_fn (function): objective function to minimize

    Returns:
        array: final loss values for each parameter set
    """

    run_fn = lambda p: run_adam_loop(p, adam_lr, adam_b1, adam_b2, num_steps, objective_fn)
    return jax.vmap(run_fn)(param_batch)
