import jax
import optax as opx

def adam_step(params, opt_state, optimizer, objective_fn):
    loss, grads = jax.value_and_grad(objective_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = opx.apply_updates(params, updates)
    return new_params, opt_state, loss

def run_adam_loop(params_init, adam_lr, adam_b1, adam_b2, num_steps, objective_fn):
    optimizer = opx.adam(learning_rate=adam_lr, b1=adam_b1, b2=adam_b2)
    opt_state = optimizer.init(params_init)

    def step_fn(carry, _):
        params, opt_state = carry
        params, opt_state, _ = adam_step(params, opt_state, optimizer, objective_fn)
        return (params, opt_state), None

    (final_params, _), _ = jax.lax.scan(step_fn, (params_init, opt_state), None, length=num_steps)
    final_loss = objective_fn(final_params)
    return final_loss


# --- Batched version with vmap ---
def batch_run_adam(param_batch, adam_lr, adam_b1, adam_b2, num_steps, objective_fn):
    run_fn = lambda p: run_adam_loop(p, adam_lr, adam_b1, adam_b2, num_steps, objective_fn)
    return jax.vmap(run_fn)(param_batch)
