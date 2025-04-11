import jax
import jax.numpy as jnp
from scipy.stats import qmc
from pdu_rhs import V_RATIO

"""" 
Bounds ordered by: 

PD_internal_mult, ald_internal_mult, nadx_internal_mult,
kcatN_CDE, Km_CDE, 
kcatN_P, dG_P, Kf1_P, Kf2_P, Kr1_P, Kr2_P, 
kcatN_Q, dG_Q, Kf1_Q, Kf2_Q, Kr1_Q, Kr2_Q, 
kcatN_L,  dG_L, Kf_L, Kr_L, 
kcatN_W, dG_W, Kf_W, Kr_W, 
perm_metab, perm_nadx, 
evap_rate, vol_ratio
"""
KJ_TO_J = 1000

LB = [0., 0., 0., \
      1e3, 0., \
      1e2, -41.5 * KJ_TO_J, 0., 0., 0., 0., \
      1e2, -19.7 * KJ_TO_J, 0., 0., 0., 0., \
      10, 14.1 * KJ_TO_J, 0., 0., \
      0.1, -38.6 * KJ_TO_J, 0., 0., \
      -8., -10., -4.5, 0.5 * V_RATIO
      ]
UB = [0.5, 0.5, 1.,
      5e7, 50., \
      1e6, -36.5 * KJ_TO_J, 200., 200., 400., 400., \
      1e6, 0., 200., 200., 400., 400., \
      1e6, 27.9 * KJ_TO_J, 200., 400., \
      1e3, -26.3 * KJ_TO_J, 200., 400., \
      -2., -2., -4., 2. * V_RATIO
      ]

def lh_samples(n_samples):
    """
    Generate Latin Hypercube samples within the specified bounds.
    
    Args:
        n_samples (int): number of samples to generate

    Returns:
        array: generated samples
    """

    sampler = qmc.LatinHypercube(d=len(LB))
    sample = sampler.random(n=n_samples)
    return jnp.array(qmc.scale(sample, LB, UB)) # scale according to bounds
