import jax
import jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)
from pdu_rhs import pdu_rhs, V_EXT, V_INT

############################
#  Construct IC from data  #
############################

@jax.jit
def generate_y0(params, data_ICs):
    """
    Generate the initial conditions for the system of ODEs.

    Args:
        params (list): list of parameters for the system
        data_ICs (list): initial concentrations of [12PD, Ald, NAD, NADH] in the system
    Returns:
        array: initial conditions for the system of ODEs
    """

    data_PD_init, data_Ald_init, data_NAD_init, data_NADH_init = data_ICs
    PD_internal_mult, ald_internal_mult, nadx_internal_mult, *_  = params
    
    PD_in_init = data_PD_init * PD_internal_mult
    Ald_in_init = data_Ald_init * ald_internal_mult
    NAD_in_init = data_NAD_init * nadx_internal_mult
    NADH_in_init = data_NADH_init * nadx_internal_mult
    
    PD_ext_init = data_PD_init
    Ald_ext_init = data_Ald_init
    NAD_ext_init = data_NAD_init
    NADH_ext_init = data_NADH_init

    return jnp.array([
        PD_in_init, Ald_in_init, 0.0, 0.0, NAD_in_init, NADH_in_init, 0.0, 0.0,
        PD_ext_init, Ald_ext_init, 0.0, 0.0, NAD_ext_init, NADH_ext_init, 0.0, 0.0
    ])

##############################
#  Simulation & measurement  #
##############################

@jax.jit
def simulate_pdu(params, y0, t0, t1, dt, timepoints=None, KO=None):
    """
    Simulate the reaction system using the provided parameters and initial conditions.

    Args:
        params (list): list of parameters for the system
        y0 (array): initial conditions for the system of ODEs
        t0 (float): initial time
        t1 (float): final time
        dt (float): time step size
        timepoints (array, optional): specific time points to evaluate the solution
        KO (str, optional): knockout condition (default: None)

    Returns:
        array: concentrations of the metabolites in the system at each time point
    """
    # Define the time points for simulation
    if timepoints is None:
        timepoints = jnp.linspace(t0, t1, 500)

    solver = dfx.Kvaerno5()
    term = dfx.ODETerm(pdu_rhs)
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        max_steps=2**10,
        args=(params, KO),
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(ts=timepoints)
    )
    
    return sol.ts, sol.ys

@jax.jit  
def measure_pdu(simulation_results):
    """
    Measure the concentrations of the metabolites in the system at each time point.

    Args:
        simulation_results (array): concentrations of the metabolites in the system at each time point

    Returns:
        array: measured concentrations of the metabolites in the system at each time point
    """
    ts, ys = simulation_results

    measured_ext = ys[:, [8, 9, 11, 15]]
    measured_in = ys[:, [0, 1, 3, 7]]
    measured = (measured_ext * V_EXT + measured_in * V_INT) / (V_EXT + V_INT)
    return (ts, measured)
