import jax
import jax.numpy as jnp
from scipy.stats import qmc

LB = []
UB = []

def lh_samples(n_samples, n_dim):
    sampler = qmc.LatinHypercube(d=n_dim)
    sample = sampler.random(n=n_samples)
    return jnp.array(qmc.scale(sample, LB, UB))