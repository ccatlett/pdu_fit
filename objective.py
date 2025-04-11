import jax
import jax.numpy as jnp
from functools import partial
from simulation import simulate_pdu, measure_pdu, generate_y0

###################
#  Evaluate loss  #
###################

@partial(jax.jit, static_argnums=(0,6))
def objective(parameterized_loss_fn, params, x, t, dt, data_ICs, KO):
    """
    Objective function to minimize the loss between the model output and the target data.
    
    Args:
        parameterized_loss_fn (function): parameterized loss function
        params (array): parameters for the model
        data (tuple): tuple containing experimental data
            - t (array): time data
            - x (array): experimental data
            - data_ICs (array): initial conditions from data
            - KO (str): knockout condition (default: None)

    Returns:
        float: computed loss value
    """

    y0 = generate_y0(params, data_ICs)
    sol = simulate_pdu(params, y0, t[0], t[-1], dt, timepoints=t, KO=KO)
    _, y_meas = measure_pdu(sol)
    return parameterized_loss_fn(x, y_meas)

################################
#  Construct loss via options  #
################################

def parameterize_loss(x, t, loss_fn):
    """
    Parameterize the loss function with the given data to make weights
    
    Args:
        x (array): input data
        t (array): time data
        loss_fn (function): loss function to parameterize

    Returns:
        function: parameterized loss function
    """

    return loss_fn(x, t)

def construct_loss(options):
    """
    Construct the unparameterized loss function based on the provided options.

    Args:
        options (dict): dictionary containing the options for the loss function
            - "loss_fn": type of loss function to use (e.g., "L2", "MSE")
            - "t_weight": weighting in time (e.g., "inv_stepsize", "logarithmic", "linear")
            - "weight": weighting by state (e.g., "variance", "range", "relative", "max_relative_scale")
    
    Returns:
        function: loss function (not yet parameterized)
    """
    # Choose function type
    match options["loss_fn"]:
        case "L2":
            loss_fn = L2
        case "MSE":
            loss_fn = MSE

    # Choose weighting in time
    match options["t_weight"]:
        case "inv_stepsize":
            t_weights = inv_stepsize
        case "logarithmic":
            t_weights = lambda t : jnp.log(jnp.linspace(1, t[-1], len(t))) + 1
        case "logarithmic_stepsize":
            t_weights = lambda t : jnp.log(t + 1) + 1
        case "linear":
            t_weights = lambda t : jnp.linspace(1, t[-1], len(t)) + 1
        case _:
            t_weights = lambda t : jnp.ones(len(t))
    
    # Choose weighting by state
    match options["weight"]:
        case "variance":
            weights = lambda x : 1 / (jnp.var(x, axis=0) + 1e-6)
        case "range":
            weights = lambda x : 1 / (jnp.ptp(x, axis=0) + 1e-6)
        case "relative":
            weights = lambda x : 1 / (jnp.mean(jnp.abs(x), axis=0) + 1e-6)
        case "max_relative_scale":
            weights = max_relative_values
        case _:
            weights = lambda x : jnp.ones(x.shape[1])

    # Eval return function
    return lambda x, t: loss_fn(weights(x), t_weights(t))

###################################
#  Helpers for constructing loss  #
###################################

def MSE(weights, t_weights):
    """
    Mean Squared Error (MSE) loss function with weights.
    
    Args:
        weights (array): weights for each state variable
        t_weights (array): weights for each time point  
        
    Returns:
        function: MSE loss function (fully parameterized)
    """
    # Normalize weights to sum to 1
    weights = weights / jnp.sum(weights)
    t_weights = t_weights / jnp.sum(t_weights)

    return lambda x, y : jnp.sum(((x - y)**2 * weights) * t_weights[:, jnp.newaxis])

def L2(weights, t_weights):
    """
    L2 loss function with weights.
    
    Args:
        weights (array): weights for each state variable
        t_weights (array): weights for each time point  
        
    Returns:
        function: L2 loss function (fully parameterized)
    """
    # Normalize weights to sum to 1
    weights = weights / jnp.sum(weights)
    t_weights = t_weights / jnp.sum(t_weights)

    return lambda x, y : MSE(weights, t_weights)(x, y) * (x.shape[0] * x.shape[1])

def inv_stepsize(t):
    """
    Inverse step size weighting function for time data.
    
    Args:
        t (array): time data

    Returns:
        array: inverse step size weights
    """
    first = t[1] - t[0]
    return jnp.insert(jnp.diff(t), 0, first)

def max_relative_values(x):
    """
    Calculate the maximum relative values for each state variable and make weights
    
    Args:
        x (array): input data

    Returns:
        array: maximum relative values for each state variable
    """
    max_vals = jnp.max(jnp.abs(x), axis=0)
    max_overall = jnp.max(max_vals)
    return max_overall / (max_vals + 1e-6)
