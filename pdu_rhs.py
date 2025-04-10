import jax
import jax.numpy as jnp
from scipy import constants as sciconst

#######################################
#  Experimental constant definitions  #
#######################################

TEMP = 303.15  # temp of the experiment (K)
RC = 0.69e-5 # compartment radius (cm)
NUM_MCP = 2007333333 # number of MCPs in reaction volume
V_TOT = 30*0.001 # total volume (cm3)

def scale_vol(radius):
    return 1e6/(sciconst.Avogadro*(4.*jnp.pi/3.)*radius**3)

V_INT = NUM_MCP*(4.*jnp.pi/3.)*RC**3 # internal MCP volume total (cm3)
V_EXT = V_TOT - V_INT # external volume (cm3)
V_RATIO = V_EXT/V_INT
V_SCALE = scale_vol(RC)

####################################
#  Reaction mechanism helper f'ns  #
####################################


def uni_uni(substrate, kcatN, Km):
    """
    Uni-uni irreversible Michaelis-Menten kinetics

    Args:
        substrate (float): concentration of substrate (mM) 
        kcatN (float): turnover number * # of enzymes (# s^-1)
        KM (float): Michaelis-Menten constant (mM)

    Returns:
        float: reaction rate (mM s^-1)
    """
    numer = kcatN * substrate
    denom = Km + substrate
    return jnp.divide(numer, denom) * V_SCALE


def uni_uni_rev(substrate, product, dG, kcatN, Kf, Kr):
    """
    Uni-uni reversible Michaelis-Menten kinetics

    Args:
        substrate (float): concentration of substrate (mM) 
        product (float): concentration of product (mM)
        dG (float): change in free energy (J mol^-1 K^-1)
        kcatN (float): turnover number * # of enzymes (# s^-1)
        KF (float): forward Michaelis constant (mM)
        KR (float): reverse Michaelis constant (mM)
        
    Returns:
        float: reaction rate (mM s^-1)
    """
    thermo = jnp.exp(-dG / (sciconst.R * TEMP))
    numer = kcatN * Kr * (substrate - thermo * product)
    denom = Kf * Kr + Kr * substrate + Kf * product
    return jnp.divide(numer, denom) * V_SCALE


def bi_bi_rev(substrate_1, substrate_2, product_1, product_2,\
               dG, kcatN, Kf_1, Kf_2, Kr_1, Kr_2):
    """
    Uni-uni reversible Michaelis-Menten kinetics

    Args:
        substrate_1 (float): concentration of substrate 1 (mM) 
        substrate_2 (float): concentration of substrate 1 (mM) 
        product_1 (float): concentration of product 1 (mM)
        product_2 (float): concentration of product 1 (mM)
        dG (float): change in free energy (J mol^-1 K^-1)
        kcatN (float): turnover number * # of enzymes (# s^-1)
        KF_1 (float): forward Michaelis constant 1 (mM)
        KF_2 (float): forward Michaelis constant 2 (mM)
        KR_1 (float): reverse Michaelis constant 1 (mM)
        KR_2 (float): reverse Michaelis constant 2 (mM)
        
    Returns:
        float: reaction rate (mM s^-1)
    """

    thermo = jnp.exp(-dG / (sciconst.R * TEMP))
    numer = kcatN * Kr_1 * Kr_2 * (substrate_1 * substrate_2 \
                                    - thermo * product_1 * product_2)
    denom = (Kf_1 * Kf_2 * Kr_1 * Kr_2) \
          + (Kr_1 * Kr_2 * substrate_1 * substrate_2) \
          + (Kf_1 * Kf_2 * product_1 * product_2)
    return jnp.divide(numer, denom) * V_SCALE


def evap(substrate, evap_rate):
    """
    Evaporation reaction

    Args:
        substrate (float): concentration of substrate (mM) 
        rate (float): log10 evaporation rate (mM/s)
        
    Returns:
        float: reaction rate (mM s^-1)
    """

    return jnp.power(10, evap_rate) * substrate 


def i_perm(substrate_in, substrate_ext, k):
    """
    Diffusion reaction into MCP

    Args:
        substrate_in (float): concentration of substrate in (mM) 
        substrate_ext (float): concentration of substrate out (mM) 
        k (float): log10 permeability constant/diffusion velocity (cm s^-1)
        
    Returns:
        float: reaction rate (mM s^-1)
    """

    return jnp.divide(3. * jnp.power(10, k) * (substrate_ext - substrate_in), RC)


def e_perm(substrate_in, substrate_ext, k, vol_ratio):
    """
    Diffusion reaction into MCP

    Args:
        substrate_in (float): concentration of substrate in (mM) 
        substrate_ext (float): concentration of substrate out (mM) 
        k (float): log10 permeability constant/diffusion velocity (cm s^-1)
        
    Returns:
        float: reaction rate (mM s^-1)
    """

    return jnp.divide(3. * jnp.power(10, k) * (substrate_in - substrate_ext), RC * vol_ratio)

######################
#  Full Pdu RHS f'n  #
######################

@jax.jit
def pdu_rhs(t, y, args):
    """
    Function to calculate the right-hand side of the Pdu reaction system.
    Args:
        t (float): time (s)
        y (array): concentrations of the metabolites in the system
        args (tuple): parameters for the reaction system
        KO (str): knockout condition (default: None)

    Returns:
        array: rates of change of the metabolites (mM s^-1)
    """
    params, KO = args

    _, _, _, \
    kcatN_CDE, Km_CDE, \
    kcatN_P, dG_P, Kf1_P, Kf2_P, Kr1_P, Kr2_P, \
    kcatN_Q, dG_Q, Kf1_Q, Kf2_Q, Kr1_Q, Kr2_Q, \
    kcatN_L,  dG_L, Kf_L, Kr_L, \
    kcatN_W, dG_W, Kf_W, Kr_W, \
    perm_metab, perm_nadx, \
    evap_rate, vol_ratio = params # first 3 only in IC creation

    PD_in, Ald_in, CoA_in, Propanol_in, NAD_in, NADH_in, PO4_in, Propionate_in, \
    PD_ext, Ald_ext, CoA_ext, Propanol_ext, NAD_ext, NADH_ext, PO4_ext, Propionate_ext = y

    # Define enzyme fluxes using helper functions (internal: CDE, P, Q, L; external: W)

    CDE_flux = uni_uni(PD_in, kcatN_CDE, Km_CDE)

    match KO: # handle knockout conditions for dPduP, dPduQ, dPduPQ
        case 'dP': 
            P_flux = 0.0
            Q_flux = bi_bi_rev(Ald_in, NADH_in, Propanol_in, NAD_in, dG_Q, kcatN_Q, Kf1_Q, Kf2_Q, Kr1_Q, Kr2_Q)
        case 'dQ':
            P_flux = bi_bi_rev(Ald_in, NAD_in, CoA_in, NADH_in, dG_P, kcatN_P,  Kf1_P, Kf2_P, Kr1_P, Kr2_P)
            Q_flux = 0.0
        case 'dPQ':
            P_flux = 0.0
            Q_flux = 0.0
        case _:
            P_flux = bi_bi_rev(Ald_in, NAD_in, CoA_in, NADH_in, dG_P, kcatN_P,  Kf1_P, Kf2_P, Kr1_P, Kr2_P)
            Q_flux = bi_bi_rev(Ald_in, NADH_in, Propanol_in, NAD_in, dG_Q, kcatN_Q, Kf1_Q, Kf2_Q, Kr1_Q, Kr2_Q)
    
    L_flux = uni_uni_rev(CoA_in, PO4_in, dG_L, kcatN_L, Kf_L, Kr_L)
    W_flux = uni_uni_rev(PO4_ext, Propionate_ext, dG_W, kcatN_W, Kf_W, Kr_W)

    # Define diffusion fluxes using helper functions (internal: i_perm; external: e_perm)

    PD_diff_out = i_perm(PD_in, PD_ext, perm_metab)
    Ald_diff_out = i_perm(Ald_in, Ald_ext, perm_metab)
    CoA_diff_out = i_perm(CoA_in, CoA_ext, perm_metab)
    Propanol_diff_out = i_perm(Propanol_in, Propanol_ext, perm_metab)
    NAD_diff_out = i_perm(NAD_in, NAD_ext, perm_nadx)
    NADH_diff_out = i_perm(NADH_in, NADH_ext, perm_nadx)
    PO4_diff_out = i_perm(PO4_in, PO4_ext, perm_metab)
    Propionate_diff_out = i_perm(Propionate_in, Propionate_ext, perm_metab)

    PD_diff_in = e_perm(PD_in, PD_ext, perm_metab, vol_ratio)
    Ald_diff_in = e_perm(Ald_in, Ald_ext, perm_metab, vol_ratio)
    CoA_diff_in = e_perm(CoA_in, CoA_ext, perm_metab, vol_ratio)
    Propanol_diff_in = e_perm(Propanol_in, Propanol_ext, perm_metab, vol_ratio)
    NAD_diff_in = e_perm(NAD_in, NAD_ext, perm_nadx, vol_ratio)
    NADH_diff_in = e_perm(NADH_in, NADH_ext, perm_nadx, vol_ratio)
    PO4_diff_in = e_perm(PO4_in, PO4_ext, perm_metab, vol_ratio)
    Propionate_diff_in = e_perm(Propionate_in, Propionate_ext, perm_metab, vol_ratio)

    evap_flux = evap(Ald_ext, evap_rate)

    # Define RHS

    dPD_in = - CDE_flux + PD_diff_out # internal values
    dAld_in = CDE_flux - P_flux - Q_flux + Ald_diff_out
    dCoA_in = P_flux - L_flux + CoA_diff_out
    dPropanol_in = Q_flux + Propanol_diff_out
    dNAD_in = - P_flux + Q_flux + NAD_diff_out
    dNADH_in = P_flux - Q_flux + NADH_diff_out
    dPO4_in = L_flux + PO4_diff_out
    dPropionate_in = Propionate_diff_out

    dPD_ext = PD_diff_in # external values
    dAld_ext = Ald_diff_in - evap_flux
    dCoA_ext = CoA_diff_in
    dPropanol_ext = Propanol_diff_in
    dNAD_ext = NAD_diff_in
    dNADH_ext = NADH_diff_in
    dPO4_ext = - W_flux + PO4_diff_in
    dPropionate_ext = W_flux + Propionate_diff_in

    return jnp.array([dPD_in, dAld_in, dCoA_in, dPropanol_in, dNAD_in, \
                      dNADH_in, dPO4_in, dPropionate_in, \
                      dPD_ext, dAld_ext, dCoA_ext, dPropanol_ext, \
                      dNAD_ext, dNADH_ext, dPO4_ext, dPropionate_ext])
