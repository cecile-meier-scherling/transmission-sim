from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from dl4bi_sps.malaria import (
    MalariaHistogramState,
    MalariaTransmission,
    _host_state_table_jax,
    _next_state_probs,
)


def mean_allele_frequency(
    counts: Array,
    state_num_mutants: Array,
    state_num_alleles: Array,
) -> Array:
    """Compute population mean allele frequency from host-state histograms."""
    counts = counts.astype(jnp.float32)
    state_allele_freq = (
        state_num_mutants.astype(jnp.float32) / state_num_alleles.astype(jnp.float32)
    )
    num_hosts = jnp.sum(counts, axis=-1)
    weighted_sum = jnp.sum(counts * state_allele_freq[None, ...], axis=-1)
    return weighted_sum / num_hosts


def prevalence(counts: Array, state_num_mutants: Array) -> Array:
    """Compute prevalence (fraction of hosts with >=1 mutant allele)."""
    counts = counts.astype(jnp.float32)
    infected_mask = (state_num_mutants > 0).astype(jnp.float32)
    num_hosts = jnp.sum(counts, axis=-1)
    infected_hosts = jnp.sum(counts * infected_mask[None, ...], axis=-1)
    return infected_hosts / num_hosts


def lineage_expectation_formula(
    Q: int | float,
    sw: Array | float,
    st: Array | float,
    chi: Array | float,
) -> Array:
    """Approximate E[M_{t+1}|M_t=1] from the user-supplied expression."""
    q = jnp.asarray(Q, dtype=jnp.float32)
    sw = jnp.asarray(sw, dtype=jnp.float32)
    st = jnp.asarray(st, dtype=jnp.float32)
    chi = jnp.asarray(chi, dtype=jnp.float32)
    within_host = q * (1.0 + sw) / (q + sw)
    transmission = 1.0 + st / q
    superinfection = 1.0 + chi
    return within_host * transmission * superinfection


def initial_state_m_equals_one(
    chi: float,
    sw: float,
    st: float,
    num_hosts: int,
    Q: int,
) -> MalariaHistogramState:
    """Create a state with one mutant lineage in the whole population."""
    if num_hosts < 2:
        raise ValueError("num_hosts must be at least 2 for distinct superinfection parents.")
    state_num_mutants, state_num_alleles = _host_state_table_jax(int(Q))
    num_states = state_num_mutants.shape[0]
    num_hosts_by_state = jnp.zeros((1, num_states), dtype=jnp.int32)
    num_hosts_by_state = num_hosts_by_state.at[:, 0].set(int(num_hosts) - 1)
    num_hosts_by_state = num_hosts_by_state.at[:, 1].set(1)
    return MalariaHistogramState(
        chi=jnp.asarray([chi], dtype=jnp.float32),
        sw=jnp.asarray([sw], dtype=jnp.float32),
        st=jnp.asarray([st], dtype=jnp.float32),
        p_init=jnp.asarray([1.0 / float(Q)], dtype=jnp.float32),
        num_hosts=jnp.asarray([int(num_hosts)], dtype=jnp.int32),
        num_hosts_by_state=num_hosts_by_state,
        state_num_mutants=state_num_mutants,
        state_num_alleles=state_num_alleles,
        num_transmitted_alleles=int(Q),
    )


def exact_lineage_expectation(state: MalariaHistogramState) -> Array:
    """Compute exact one-step total E[M_{t+1}|state] from transition probabilities."""
    state_probs = _next_state_probs(state)
    expected_mutants_per_host = jnp.sum(
        state_probs * state.state_num_mutants.astype(jnp.float32)[None, :],
        axis=-1,
    )
    return state.num_hosts.astype(jnp.float32) * expected_mutants_per_host


def run_trajectory(
    sim: MalariaTransmission,
    rng: Array,
    num_generations: int,
    num_simulations: int,
    num_transmitted_alleles: int,
    p_init: float,
) -> tuple[Array, Array]:
    """Simulate and return mean allele-frequency and prevalence trajectories."""
    num_hosts_by_state, state_num_mutants, state_num_alleles, _ = sim.simulate(
        rng,
        num_generations=num_generations,
        num_simulations=num_simulations,
        num_transmitted_alleles=num_transmitted_alleles,
        p_init=p_init,
    )
    mean_freq = mean_allele_frequency(
        num_hosts_by_state,
        state_num_mutants,
        state_num_alleles,
    )
    prev = prevalence(num_hosts_by_state, state_num_mutants)
    return mean_freq, prev
