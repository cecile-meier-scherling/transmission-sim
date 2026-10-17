from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, random
from jax.scipy.special import gammaln

from .priors import Prior


@dataclass(frozen=True)
class MalariaHistogramState:
    """Malaria state represented as host-state counts."""

    chi: Array
    sw: Array
    st: Array
    p_init: Array
    num_hosts: Array
    num_hosts_by_state: Array
    state_num_mutants: Array
    state_num_alleles: Array
    num_transmitted_alleles: int

    @staticmethod
    def parameter_names() -> tuple[str, ...]:
        return (
            "num_hosts",
            "num_transmitted_alleles",
            "chi",
            "sw",
            "st",
            "p_init",
        )

    def parameter_matrix(self) -> tuple[Array, tuple[str, ...]]:
        """Return sampled parameters as `[num_simulations, parameter]` rows."""
        return (
            jnp.stack(
                (
                    self.num_hosts.astype(jnp.float32),
                    jnp.full(
                        (self.chi.shape[0],),
                        self.num_transmitted_alleles,
                        dtype=jnp.float32,
                    ),
                    self.chi,
                    self.sw,
                    self.st,
                    self.p_init,
                ),
                axis=-1,
            ),
            self.parameter_names(),
        )


@dataclass(frozen=True)
class MalariaSimulationBatch:
    """Complete histogram simulation output for one static-shape batch."""

    num_hosts_by_state: Array
    state_num_mutants: Array
    state_num_alleles: Array
    parameters: Array
    parameter_names: tuple[str, ...]
    num_transmitted_alleles: int
    batch_p_init: float

jax.tree_util.register_pytree_node(
    MalariaHistogramState,
    lambda d: (
        (
            d.chi,
            d.sw,
            d.st,
            d.p_init,
            d.num_hosts,
            d.num_hosts_by_state,
            d.state_num_mutants,
            d.state_num_alleles,
        ),
        d.num_transmitted_alleles,
    ),
    lambda q, children: MalariaHistogramState(
        *children,
        num_transmitted_alleles=q,
    ),
)


@dataclass
class MalariaTransmission:
    """Forward-in-time malaria parasite transmission simulator.

    Args:
        num_hosts: Prior for the effective number of hosts.
        num_transmitted_alleles: Prior for alleles transmitted per vector bite.
        chi: Prior over superinfection probabilities.
        sw: Prior over within-host selection coefficients.
        st: Prior over transmission selection coefficients.
        p_init: Prior over founding mutant allele frequencies.
    """

    num_hosts: Prior = Prior("fixed", {"value": 1024})
    num_transmitted_alleles: Prior = Prior("randint", {"minval": 1, "maxval": 10})
    chi: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})
    sw: Prior = Prior("uniform", {"minval": -0.999, "maxval": 0.0})
    st: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})
    p_init: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})

    def sample_batch_params(self, rng: Array) -> tuple[int, float]:
        """Sample batch-level `(num_transmitted_alleles, p_init)` values."""
        rng_q, rng_p_init = random.split(rng, 2)
        num_transmitted_alleles = int(
            self.num_transmitted_alleles.sample(rng_q, (1,))[0]
        )
        p_init = float(self.p_init.sample(rng_p_init, (1,))[0])
        return num_transmitted_alleles, p_init

    def initial_state(
        self,
        rng: Array,
        num_simulations: int,
        num_transmitted_alleles: int,
        p_init: float,
    ) -> MalariaHistogramState:
        """Sample per-simulation parameters and initialise founding host-state counts."""
        if num_transmitted_alleles < 1:
            raise ValueError("num_transmitted_alleles must be at least 1.")

        rng_num_hosts, rng_chi, rng_sw, rng_st, rng_founder = random.split(rng, 5)
        num_hosts = self.num_hosts.sample(rng_num_hosts, (num_simulations,)).astype(
            jnp.int32
        )
        num_hosts = jnp.maximum(num_hosts, 2)
        chi = self.chi.sample(rng_chi, (num_simulations,)).astype(jnp.float32)
        sw = self.sw.sample(rng_sw, (num_simulations,)).astype(jnp.float32)
        st = self.st.sample(rng_st, (num_simulations,)).astype(jnp.float32)
        p_init_arr = jnp.full((num_simulations,), p_init, dtype=jnp.float32)

        state_num_mutants, state_num_alleles = _host_state_table_jax(
            num_transmitted_alleles
        )
        founder_probs = jnp.concatenate(
            [
                _binomial_pmf(num_transmitted_alleles, p_init_arr),
                jnp.zeros((num_simulations, 2 * num_transmitted_alleles + 1)),
            ],
            axis=-1,
        )
        num_hosts_by_state = _sample_multinomial_counts(
            rng_founder,
            num_hosts,
            founder_probs,
        )
        return MalariaHistogramState(
            chi,
            sw,
            st,
            p_init_arr,
            num_hosts,
            num_hosts_by_state,
            state_num_mutants,
            state_num_alleles,
            int(num_transmitted_alleles),
        )

    def simulate(
        self,
        rng: Array,
        num_generations: int,
        num_simulations: int,
        num_transmitted_alleles: int,
        p_init: float,
    ) -> tuple[Array, Array, Array, MalariaHistogramState]:
        """Simulate host-state count trajectories.

        Batch-level parameters ``num_transmitted_alleles`` and ``p_init`` are
        shared across all simulations. ``chi``, ``num_hosts``, ``sw``, and ``st`` are
        sampled independently per simulation.

        Returns:
            Tuple of host-state counts with shape
            `[num_simulations, num_generations, state]`, state mutant counts
            with shape `[state]`, state allele counts with shape `[state]`,
            and the final simulator state.
        """
        if num_generations < 1:
            raise ValueError("num_generations must be at least 1.")
        rng_init, rng_sim = random.split(rng)
        state = self.initial_state(
            rng_init,
            num_simulations,
            num_transmitted_alleles,
            p_init,
        )
        return _simulate_jitted(rng_sim, state, num_generations)

    def simulate_batch(
        self,
        rng: Array,
        num_generations: int,
        num_simulations: int,
        num_transmitted_alleles: int,
        p_init: float,
    ) -> MalariaSimulationBatch:
        """Simulate and package a complete static-shape histogram batch."""
        num_hosts_by_state, state_num_mutants, state_num_alleles, state = self.simulate(
            rng,
            num_generations=num_generations,
            num_simulations=num_simulations,
            num_transmitted_alleles=num_transmitted_alleles,
            p_init=p_init,
        )
        parameters, parameter_names = state.parameter_matrix()
        return MalariaSimulationBatch(
            num_hosts_by_state=num_hosts_by_state,
            state_num_mutants=state_num_mutants,
            state_num_alleles=state_num_alleles,
            parameters=parameters,
            parameter_names=parameter_names,
            num_transmitted_alleles=int(num_transmitted_alleles),
            batch_p_init=float(p_init),
        )


@partial(jit, static_argnames=("num_generations",))
def _simulate_jitted(
    rng: Array,
    state: MalariaHistogramState,
    num_generations: int,
) -> tuple[Array, Array, Array, MalariaHistogramState]:
    """Run the jitted simulator loop."""
    if num_generations == 1:
        return (
            state.num_hosts_by_state[:, None, :],
            state.state_num_mutants,
            state.state_num_alleles,
            state,
        )

    rng_steps = random.split(rng, num_generations - 1)

    def step(carry: MalariaHistogramState, rng_step: Array):
        next_state = _transmit_generation(rng_step, carry)
        return next_state, next_state.num_hosts_by_state

    final_state, histogram_steps = lax.scan(step, state, rng_steps)
    histogram_steps = jnp.moveaxis(histogram_steps, 0, 1)
    num_hosts_by_state = jnp.concatenate(
        [state.num_hosts_by_state[:, None, :], histogram_steps],
        axis=1,
    )
    return (
        num_hosts_by_state,
        state.state_num_mutants,
        state.state_num_alleles,
        final_state,
    )


def _transmit_generation(
    rng: Array,
    state: MalariaHistogramState,
) -> MalariaHistogramState:
    state_probs = _next_state_probs(state)
    num_hosts_by_state = _sample_multinomial_counts(
        rng,
        state.num_hosts,
        state_probs,
    )
    return MalariaHistogramState(
        state.chi,
        state.sw,
        state.st,
        state.p_init,
        state.num_hosts,
        num_hosts_by_state,
        state.state_num_mutants,
        state.state_num_alleles,
        state.num_transmitted_alleles,
    )


def _next_state_probs(state: MalariaHistogramState) -> Array:
    num_transmitted_alleles = state.num_transmitted_alleles
    state_allele_freq = (
        state.state_num_mutants.astype(jnp.float32)
        / state.state_num_alleles.astype(jnp.float32)
    )
    host_weights = 1.0 + state.st[:, None] * state_allele_freq[None, :]
    weighted_hosts = state.num_hosts_by_state.astype(jnp.float32) * host_weights
    parent_probs = weighted_hosts / jnp.sum(weighted_hosts, axis=-1, keepdims=True)

    parent_pool_kernel = _binomial_pmf(
        num_transmitted_alleles,
        state_allele_freq,
    )
    single_parent_pool_probs = parent_probs @ parent_pool_kernel
    non_superinfected_probs = _post_host_process_probs(
        single_parent_pool_probs,
        num_transmitted_alleles,
        state.sw,
    )

    # Draw the first host with transmission weights, then exclude that host
    # before drawing the second. Distinct hosts can share the same state.
    remaining_counts = jnp.maximum(
        state.num_hosts_by_state[:, None, :]
        - jnp.eye(parent_probs.shape[-1])[None, :, :],
        0.0,
    )
    second_weights = remaining_counts * host_weights[:, None, :]
    second_probs = second_weights / jnp.sum(second_weights, axis=-1, keepdims=True)
    second_pool_probs = second_probs @ parent_pool_kernel
    pair_pool_probs = jax.vmap(
        jax.vmap(lambda first, second: jnp.convolve(first, second, mode="full")),
        in_axes=(None, 0),
    )(parent_pool_kernel, second_pool_probs)
    superinfected_pool_probs = jnp.einsum("bs,bsk->bk", parent_probs, pair_pool_probs)
    superinfected_probs = _post_host_process_probs(
        superinfected_pool_probs,
        2 * num_transmitted_alleles,
        state.sw,
    )
    state_probs = jnp.concatenate(
        [
            (1.0 - state.chi[:, None]) * non_superinfected_probs,
            state.chi[:, None] * superinfected_probs,
        ],
        axis=-1,
    )
    state_probs = jnp.clip(state_probs, 0.0, 1.0)
    return state_probs / jnp.sum(state_probs, axis=-1, keepdims=True)


def _sample_multinomial_counts(rng: Array, num_trials: Array, probs: Array) -> Array:
    """Sample multinomial counts with exact row sums.

    Conditional binomial draws are a factorization of multinomial sampling that
    preserves integer row sums by assigning the final category the exact
    remaining count.
    """
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    num_categories = probs.shape[-1]
    num_trials = jnp.asarray(num_trials, dtype=jnp.float32)
    if num_trials.shape != probs.shape[:-1]:
        num_trials = jnp.broadcast_to(num_trials, probs.shape[:-1])
    if num_categories == 1:
        return num_trials.astype(jnp.int32)[..., None]

    keys = random.split(rng, num_categories - 1)
    category_probs = jnp.moveaxis(probs[..., :-1], -1, 0)
    initial_remaining_count = num_trials
    initial_remaining_prob = jnp.ones(probs.shape[:-1], dtype=jnp.float32)

    def step(carry, inputs):
        remaining_count, remaining_prob = carry
        key, category_prob = inputs
        conditional_prob = _probability(
            jnp.where(remaining_prob > 0.0, category_prob / remaining_prob, 0.0)
        )
        draw = random.binomial(key, remaining_count, conditional_prob)
        draw = jnp.minimum(jnp.rint(draw), remaining_count)
        return (
            remaining_count - draw,
            jnp.maximum(remaining_prob - category_prob, 0.0),
        ), draw.astype(jnp.int32)

    (remaining_count, _), draws = lax.scan(
        step,
        (initial_remaining_count, initial_remaining_prob),
        (keys, category_probs),
    )
    draws = jnp.moveaxis(draws, 0, -1)
    final_draw = jnp.rint(remaining_count).astype(jnp.int32)[..., None]
    return jnp.concatenate([draws, final_draw], axis=-1)


def _post_host_process_probs(
    pre_process_probs: Array,
    num_alleles: int,
    sw: Array,
) -> Array:
    selection_kernel = _selection_kernel(num_alleles, sw)
    return jnp.einsum(
        "bm,bms->bs",
        pre_process_probs,
        selection_kernel,
    )


def _selection_kernel(num_alleles: int, sw: Array) -> Array:
    num_mutants = jnp.arange(num_alleles + 1, dtype=jnp.float32)
    num_wild = num_alleles - num_mutants
    mutant_weight = (1.0 + sw[:, None]) * num_mutants[None, :]
    total_weight = num_wild[None, :] + mutant_weight
    p = _probability(jnp.where(total_weight > 0.0, mutant_weight / total_weight, 0.0))
    resample_kernel = _binomial_pmf(num_alleles, p)
    identity_kernel = jnp.broadcast_to(
        jnp.eye(num_alleles + 1, dtype=jnp.float32),
        resample_kernel.shape,
    )
    return jnp.where(sw[:, None, None] == 0.0, identity_kernel, resample_kernel)


def _binomial_pmf(num_trials: int, p: Array) -> Array:
    k = jnp.arange(num_trials + 1, dtype=jnp.float32)
    log_coeff = (
        gammaln(num_trials + 1.0)
        - gammaln(k + 1.0)
        - gammaln(num_trials - k + 1.0)
    )
    p = _probability(p)
    return (
        jnp.exp(log_coeff)
        * jnp.power(p[..., None], k)
        * jnp.power(1.0 - p[..., None], num_trials - k)
    )


def _probability(value: Array) -> Array:
    return jnp.clip(value, 0.0, 1.0)


def _host_state_table(
    num_transmitted_alleles: int,
    array_module,
    dtype,
):
    base_mutants = array_module.arange(num_transmitted_alleles + 1, dtype=dtype)
    super_mutants = array_module.arange(2 * num_transmitted_alleles + 1, dtype=dtype)
    state_num_mutants = array_module.concatenate([base_mutants, super_mutants])
    state_num_alleles = array_module.concatenate(
        [
            array_module.full(
                (num_transmitted_alleles + 1,),
                num_transmitted_alleles,
                dtype=dtype,
            ),
            array_module.full(
                (2 * num_transmitted_alleles + 1,),
                2 * num_transmitted_alleles,
                dtype=dtype,
            ),
        ]
    )
    return state_num_mutants, state_num_alleles


def host_state_table(num_transmitted_alleles: int) -> tuple[np.ndarray, np.ndarray]:
    """Return host-state coordinate values for a transmission bottleneck size."""
    return _host_state_table(num_transmitted_alleles, np, np.uint16)


def _host_state_table_jax(num_transmitted_alleles: int) -> tuple[Array, Array]:
    """Return JAX host-state coordinate values for histogram construction."""
    return _host_state_table(num_transmitted_alleles, jnp, jnp.int32)
