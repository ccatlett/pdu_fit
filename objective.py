import jax.numpy as jnp

def compute_loss(measured_candidate, measured_data, options):

    match options["func"]:
        case "L2":
            loss_func = lambda x, y: jnp.sum((x - y) ** 2)
        case "LL":
            loss_func = lambda x, y, alpha: ((x - y)**2) \
              / (2 * alpha * jnp.abs(y)) + 0.5 * jnp.log(jnp.abs(y))
        case "MSE":
            loss_func = lambda x, y: jnp.mean((x - y) ** 2)
            
    match options["t_weight"]:
        case "inv_stepsize":
            t_weights = jnp.array([.5, .5, .5, .5, .5, .5, .5, .5, \
                                 .5, .5, .5, .5, .5, 18.]) / (13./2. + 18.) # normalized
        case "logarithmic":
            t_weights = jnp.log(jnp.arange(1, len(measured_candidate) + 1)) + 1 \
                / jnp.sum(jnp.log(jnp.arange(1, len(measured_candidate) + 1)) + 1)
        case None:
            t_weights = jnp.ones(len(measured_candidate))

    
    match options["weight"]:
        case "relative":
            weights = 1 / (measured_candidate + 1e-6)
        case "scaled":
            pass # TODO
        case ""
        case None:
            weights = jnp.ones(len(measured_candidate))

    