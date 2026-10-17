import numpy as np
import pytest
from jax import random
from omegaconf import OmegaConf

from dl4bi_sps.malaria import (
    MalariaHistogramState,
    MalariaSimulationBatch,
    MalariaTransmission,
    host_state_table,
)
from dl4bi_sps.priors import Prior


def test_malaria_shapes_and_bounds():
    """Verify malaria histogram trajectories have expected shape and bounds."""
    rng = random.key(42)
    sim = MalariaTransmission(num_hosts=Prior("fixed", {"value": 16}))
    num_hosts_by_state, state_num_mutants, state_num_alleles, state = sim.simulate(
        rng,
        num_generations=5,
        num_simulations=4,
        num_transmitted_alleles=2,
        p_init=0.1,
    )

    num_hosts_by_state = np.asarray(num_hosts_by_state)
    assert num_hosts_by_state.shape == (4, 5, 8)
    assert state_num_mutants.shape == (8,)
    assert state_num_alleles.shape == (8,)
    assert state.num_hosts_by_state.shape == (4, 8)
    assert state.num_transmitted_alleles == 2
    np.testing.assert_array_equal(
        num_hosts_by_state.sum(axis=-1),
        np.full((4, 5), 16),
    )


def test_malaria_deterministic_repeat():
    """The same key and configuration should reproduce identical histograms."""
    rng = random.key(123)
    sim = MalariaTransmission(num_hosts=Prior("fixed", {"value": 12}))
    hist_a, _, _, state_a = sim.simulate(
        rng,
        num_generations=4,
        num_simulations=3,
        num_transmitted_alleles=2,
        p_init=0.2,
    )
    hist_b, _, _, state_b = sim.simulate(
        rng,
        num_generations=4,
        num_simulations=3,
        num_transmitted_alleles=2,
        p_init=0.2,
    )

    np.testing.assert_array_equal(np.asarray(hist_a), np.asarray(hist_b))
    np.testing.assert_array_equal(
        np.asarray(state_a.num_hosts_by_state),
        np.asarray(state_b.num_hosts_by_state),
    )


def test_malaria_always_starts_with_initial_generation():
    """The first returned generation is the founding histogram."""
    rng = random.key(7)
    rng_init, _ = random.split(rng)
    sim = MalariaTransmission(num_hosts=Prior("fixed", {"value": 8}))
    initial = sim.initial_state(
        rng_init,
        num_simulations=2,
        num_transmitted_alleles=2,
        p_init=0.5,
    )

    num_hosts_by_state, _, _, state = sim.simulate(
        rng,
        num_generations=1,
        num_simulations=2,
        num_transmitted_alleles=2,
        p_init=0.5,
    )

    np.testing.assert_array_equal(
        np.asarray(num_hosts_by_state[:, 0, :]),
        np.asarray(initial.num_hosts_by_state),
    )
    np.testing.assert_array_equal(
        np.asarray(state.num_hosts_by_state),
        np.asarray(initial.num_hosts_by_state),
    )


def test_malaria_fixed_priors_in_parameters():
    """Fixed priors should be reflected in the state parameter matrix."""
    sim = MalariaTransmission(
        num_hosts=Prior("fixed", {"value": 10}),
        chi=Prior("fixed", {"value": 0.25}),
        sw=Prior("fixed", {"value": -0.1}),
        st=Prior("fixed", {"value": 0.2}),
    )
    _, _, _, state = sim.simulate(
        random.key(0),
        num_generations=2,
        num_simulations=5,
        num_transmitted_alleles=3,
        p_init=0.3,
    )
    params_jax, param_names = state.parameter_matrix()
    params = np.asarray(params_jax)

    assert param_names == (
        "num_hosts",
        "num_transmitted_alleles",
        "chi",
        "sw",
        "st",
        "p_init",
    )
    assert params.shape == (5, len(param_names))
    np.testing.assert_array_equal(params[:, 0], np.full(5, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(params[:, 1], np.full(5, 3.0, dtype=np.float32))
    np.testing.assert_allclose(params[:, 2], 0.25)
    np.testing.assert_allclose(params[:, 3], -0.1)
    np.testing.assert_allclose(params[:, 4], 0.2)
    np.testing.assert_allclose(params[:, 5], 0.3)


def test_malaria_num_hosts_sampling_uses_choice_prior():
    """Per-simulation host counts can be sampled from a finite configured set."""
    sim = MalariaTransmission(num_hosts=Prior("choice", {"values": [1024, 2048]}))
    state = sim.initial_state(
        random.key(0),
        num_simulations=32,
        num_transmitted_alleles=4,
        p_init=0.1,
    )
    samples = set(np.asarray(state.num_hosts).tolist())

    assert samples <= {1024, 2048}
    assert samples


def test_malaria_batch_params_sampling():
    """Batch-level Q and p_init are sampled once per batch."""
    sim = MalariaTransmission(
        num_transmitted_alleles=Prior("choice", {"values": [2, 4]}),
        p_init=Prior("choice", {"values": [0.1, 0.5]}),
    )
    q_values = set()
    p_init_values = set()
    for seed in range(16):
        q, p_init = sim.sample_batch_params(random.key(seed))
        q_values.add(q)
        p_init_values.add(p_init)

    assert q_values <= {2, 4}
    assert all(np.isclose(p_init, [0.1, 0.5]).any() for p_init in p_init_values)


def test_malaria_zarr_smoke(tmp_path):
    """Verify the intended xarray/Zarr data can be written and read."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from examples.build_malaria_dataset import make_histogram_dataset

    sim = MalariaTransmission(num_hosts=Prior("fixed", {"value": 8}))
    batch = sim.simulate_batch(
        random.key(1),
        num_generations=3,
        num_simulations=2,
        num_transmitted_alleles=2,
        p_init=0.1,
    )
    store = tmp_path / "malaria.zarr"
    group = "num_transmitted_alleles=2"

    ds = make_histogram_dataset(
        batch,
        np.arange(2, dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=1,
    )
    ds.to_zarr(store, group=group, mode="a", consolidated=False)

    opened = xr.open_zarr(store, group=group, consolidated=False)
    assert opened["num_hosts_by_state"].shape == (2, 3, 8)
    assert opened["parameters"].shape == (2, len(batch.parameter_names))
    assert opened["num_hosts_by_state"].dtype == np.dtype("uint16")
    np.testing.assert_array_equal(
        opened["num_hosts_by_state"].sum("state").values,
        np.full((2, 3), 8, dtype=np.uint16),
    )
    num_hosts = opened["parameters"].sel(parameter="num_hosts").values
    allele_freq = (
        opened["num_hosts_by_state"] * opened["state_allele_freq"]
    ).sum("state") / num_hosts[:, None]
    assert allele_freq.shape == (2, 3)


def test_malaria_histogram_first_generation_matches_founder_distribution():
    """The first histogram generation should represent the founding population."""
    sim = MalariaTransmission(
        num_hosts=Prior("fixed", {"value": 256}),
        chi=Prior("fixed", {"value": 0.0}),
        sw=Prior("fixed", {"value": 0.0}),
        st=Prior("fixed", {"value": 0.0}),
    )
    num_hosts_by_state, _, _, _ = sim.simulate(
        random.key(1),
        num_generations=1,
        num_simulations=256,
        num_transmitted_alleles=2,
        p_init=0.25,
    )

    observed = np.asarray(num_hosts_by_state[:, 0, :]).mean(axis=0) / 256
    expected = np.asarray([0.5625, 0.375, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(observed, expected, atol=0.03)


def test_malaria_zarr_append_reuses_existing_encoding(tmp_path):
    """Appending a second batch should not resubmit encoding for existing variables."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from examples.build_malaria_dataset import _write_dataset, make_histogram_dataset

    param_names = MalariaHistogramState.parameter_names()
    state_num_mutants, state_num_alleles = host_state_table(2)
    num_hosts_by_state = np.zeros((1, 3, state_num_mutants.size), dtype=np.uint16)
    num_hosts_by_state[..., 0] = 8
    first = make_histogram_dataset(
        _batch_from_counts(num_hosts_by_state, state_num_mutants, state_num_alleles),
        np.asarray([0], dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=0,
    )
    second_num_hosts_by_state = np.zeros_like(num_hosts_by_state)
    second_num_hosts_by_state[..., 1] = 8
    second = make_histogram_dataset(
        _batch_from_counts(
            second_num_hosts_by_state,
            state_num_mutants,
            state_num_alleles,
        ),
        np.asarray([1], dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=0,
    )
    store = tmp_path / "malaria.zarr"
    group = "num_transmitted_alleles=2"
    output_cfg = OmegaConf.create(
        {
            "chunks": {
                "num_simulations": 1,
                "num_generations": 2,
                "num_states": 4,
            },
            "compression": {
                "enabled": True,
                "codec": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "shuffle",
            },
            "write_empty_chunks": False,
        }
    )

    _write_dataset(first, store, group, append=False, output_cfg=output_cfg)
    _write_dataset(second, store, group, append=True, output_cfg=output_cfg)

    opened = xr.open_zarr(store, group=group, consolidated=False)
    assert opened["num_hosts_by_state"].shape == (2, 3, state_num_mutants.size)
    assert opened["parameters"].shape == (2, len(param_names))
    np.testing.assert_array_equal(opened["simulation"].values, np.asarray([0, 1]))
    np.testing.assert_array_equal(
        opened["num_hosts_by_state"].sum("state").values,
        np.full((2, 3), 8, dtype=np.uint16),
    )
    num_hosts = opened["parameters"].sel(parameter="num_hosts").values
    allele_freq = (
        opened["num_hosts_by_state"] * opened["state_allele_freq"]
    ).sum("state") / num_hosts[:, None]
    assert allele_freq.shape == (2, 3)


def _batch_from_counts(
    num_hosts_by_state: np.ndarray,
    state_num_mutants: np.ndarray,
    state_num_alleles: np.ndarray,
) -> MalariaSimulationBatch:
    return MalariaSimulationBatch(
        num_hosts_by_state=num_hosts_by_state,
        state_num_mutants=state_num_mutants,
        state_num_alleles=state_num_alleles,
        parameters=np.asarray(
            [[8.0, 2.0, 0.1, -0.1, 0.2, 0.3]],
            dtype=np.float32,
        ),
        parameter_names=MalariaHistogramState.parameter_names(),
        num_transmitted_alleles=2,
        batch_p_init=0.3,
    )


@pytest.mark.parametrize("configured", [False, True])
def test_q_sampling_covers_one_through_nine(configured):
    if configured:
        from examples.build_malaria_dataset import _prior_from_config
        config = OmegaConf.load("examples/configs/malaria_dataset.yaml")
        sim = MalariaTransmission(
            num_transmitted_alleles=_prior_from_config(config.priors.num_transmitted_alleles)
        )
    else:
        sim = MalariaTransmission()
    samples = {sim.sample_batch_params(random.key(seed))[0] for seed in range(128)}
    assert samples == set(range(1, 10))


@pytest.mark.parametrize("q", [1, 9])
@pytest.mark.parametrize("st", [0.0, 1.0])
def test_superinfection_excludes_same_host(q, st):
    from dataclasses import replace
    import jax.numpy as jnp
    from dl4bi_sps.malaria import _next_state_probs

    sim = MalariaTransmission(
        num_hosts=Prior("fixed", {"value": 2}),
        chi=Prior("fixed", {"value": 1.0}),
        sw=Prior("fixed", {"value": 0.0}),
        st=Prior("fixed", {"value": st}),
    )
    state = sim.initial_state(random.key(0), 1, q, 0.0)
    # One all-wild host and one all-mutant host must contribute exactly q mutants.
    counts = jnp.zeros_like(state.num_hosts_by_state).at[0, 0].set(1).at[0, q].set(1)
    probs = np.asarray(_next_state_probs(replace(state, num_hosts_by_state=counts)))
    expected = np.zeros_like(probs)
    expected[0, q + 1 + q] = 1.0
    np.testing.assert_allclose(probs, expected, atol=1e-6)

    # Two distinct hosts in the same state remain valid parents.
    counts = jnp.zeros_like(counts).at[0, q].set(2)
    probs = np.asarray(_next_state_probs(replace(state, num_hosts_by_state=counts)))
    expected[:] = 0
    expected[0, -1] = 1.0
    np.testing.assert_allclose(probs, expected, atol=1e-6)


@pytest.mark.parametrize("p_init", [0.0, 1.0])
def test_no_mutation_from_fixed_population(p_init):
    sim = MalariaTransmission(num_hosts=Prior("fixed", {"value": 16}))
    counts, mutants, alleles, _ = sim.simulate(random.key(4), 3, 2, 9, p_init)
    impossible = np.asarray(mutants != (p_init * alleles))
    assert np.all(np.asarray(counts)[..., impossible] == 0)


def test_superinfection_matches_weighted_individual_parent_enumeration():
    from dataclasses import replace
    import jax.numpy as jnp
    from dl4bi_sps.malaria import _next_state_probs

    sim = MalariaTransmission(
        num_hosts=Prior("fixed", {"value": 4}),
        chi=Prior("fixed", {"value": 1.0}),
        sw=Prior("fixed", {"value": 0.0}),
        st=Prior("fixed", {"value": 0.8}),
    )
    state = sim.initial_state(random.key(0), 1, 1, 0.0)
    # Q=1 states: (0/1, 1/1, 0/2, 1/2, 2/2).
    state = replace(state, num_hosts_by_state=jnp.asarray([[2, 1, 0, 1, 0]]))
    frequencies = np.asarray([0.0, 0.0, 1.0, 0.5])
    weights = 1 + 0.8 * frequencies
    expected = np.zeros(3)
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            probability = weights[first] / weights.sum() * weights[second] / (weights.sum() - weights[first])
            pool = np.convolve([1 - frequencies[first], frequencies[first]],
                               [1 - frequencies[second], frequencies[second]])
            expected += probability * pool
    observed = np.asarray(_next_state_probs(state))[0]
    np.testing.assert_allclose(observed, np.r_[0.0, 0.0, expected], atol=1e-6)
