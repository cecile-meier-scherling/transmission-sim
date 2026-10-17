#!/usr/bin/env python3
from dataclasses import dataclass
from itertools import groupby
import shutil
from pathlib import Path

import hydra
import jax
import numpy as np
import xarray as xr
import zarr
from jax import Array, random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from zarr.codecs import BloscCodec

from dl4bi_sps.malaria import MalariaSimulationBatch, MalariaTransmission
from dl4bi_sps.priors import Prior


@dataclass(frozen=True)
class MalariaBatchSpec:
    batch_idx: int
    rng: Array
    num_transmitted_alleles: int
    p_init: float

    @property
    def group(self) -> str:
        return _shape_group(self.num_transmitted_alleles)


@hydra.main(version_base=None, config_path="configs", config_name="malaria_dataset")
def main(cfg: DictConfig) -> None:
    """Generate malaria transmission simulations and write them to Zarr."""
    store = Path(cfg.output.store)
    if store.exists():
        if cfg.output.overwrite:
            shutil.rmtree(store)
        else:
            raise FileExistsError(
                f"{store} already exists. Set output.overwrite=true to replace it."
            )
    store.parent.mkdir(parents=True, exist_ok=True)

    simulator = _build_simulator(cfg)
    seed = int(cfg.seed)
    rng = random.key(seed)
    next_sample_id = 0
    group_num_samples: dict[str, int] = {}
    group_records: list[dict] = []
    num_batches = int(cfg.num_batches)
    num_simulations_per_batch = int(cfg.num_simulations_per_batch)
    num_generations = int(cfg.num_generations)
    if num_batches < 1:
        raise ValueError("num_batches must be at least 1.")
    num_batches_per_write = int(cfg.output.num_batches_per_write)
    if num_batches_per_write < 1:
        raise ValueError("output.num_batches_per_write must be at least 1.")
    progress = tqdm(total=num_batches, desc="Generating malaria dataset", unit="batch")
    batch_specs = _sample_batch_specs(simulator, rng, num_batches)
    generation_ids = np.arange(num_generations, dtype=np.int32)

    for group, group_specs_iter in groupby(batch_specs, key=lambda spec: spec.group):
        group_specs = list(group_specs_iter)
        for write_specs in _chunks(group_specs, num_batches_per_write):
            datasets = []
            for spec in write_specs:
                ds, sample_ids = _simulate_dataset(
                    simulator,
                    spec,
                    num_generations,
                    num_simulations_per_batch,
                    seed,
                    generation_ids,
                    next_sample_id,
                )
                next_sample_id += num_simulations_per_batch
                datasets.append(ds)
                group_records.append(
                    {
                        "group": spec.group,
                        "num_transmitted_alleles": spec.num_transmitted_alleles,
                        "batch_p_init": spec.p_init,
                        "batch_idx": int(spec.batch_idx),
                        "sample_start": int(sample_ids[0]),
                        "sample_stop": int(sample_ids[-1]) + 1,
                    }
                )
                progress.update(1)
                progress.set_postfix(
                    {
                        "num_total_samples": next_sample_id,
                        "group": spec.group,
                    },
                    refresh=False,
                )
            _flush_datasets(
                datasets,
                store,
                group,
                group_num_samples,
                cfg.output,
            )

    progress.close()
    _write_root_attrs(store, cfg, group_records)
    print(f"Done. Wrote {next_sample_id} samples to {store}")


def _sample_batch_specs(
    simulator: MalariaTransmission,
    rng: Array,
    num_batches: int,
) -> list[MalariaBatchSpec]:
    """Sample all batch parameters first so same-shape batches run together."""
    specs = []
    for batch_idx in range(num_batches):
        rng, rng_batch_params = random.split(rng)
        num_transmitted_alleles, p_init = simulator.sample_batch_params(
            rng_batch_params
        )
        rng, rng_batch = random.split(rng)
        specs.append(
            MalariaBatchSpec(
                batch_idx=batch_idx,
                rng=rng_batch,
                num_transmitted_alleles=num_transmitted_alleles,
                p_init=p_init,
            )
        )
    return sorted(specs, key=lambda spec: spec.num_transmitted_alleles)


def _chunks(items: list[MalariaBatchSpec], size: int) -> list[list[MalariaBatchSpec]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _simulate_dataset(
    simulator: MalariaTransmission,
    spec: MalariaBatchSpec,
    num_generations: int,
    num_simulations: int,
    seed: int,
    generation_ids: np.ndarray,
    sample_start: int,
) -> tuple[xr.Dataset, np.ndarray]:
    sample_stop = sample_start + num_simulations
    sample_ids = np.arange(sample_start, sample_stop, dtype=np.int64)
    batch = simulator.simulate_batch(
        spec.rng,
        num_generations=num_generations,
        num_simulations=num_simulations,
        num_transmitted_alleles=spec.num_transmitted_alleles,
        p_init=spec.p_init,
    )
    ds = make_histogram_dataset(
        batch,
        sample_ids,
        generation_ids,
        seed,
        context=f"in batch {spec.batch_idx} for {spec.group}",
    )
    return ds, sample_ids


def _build_simulator(cfg: DictConfig) -> MalariaTransmission:
    priors = cfg.priors
    return MalariaTransmission(
        num_hosts=_prior_from_config(priors.num_hosts),
        num_transmitted_alleles=_prior_from_config(priors.num_transmitted_alleles),
        chi=_prior_from_config(priors.chi),
        sw=_prior_from_config(priors.sw),
        st=_prior_from_config(priors.st),
        p_init=_prior_from_config(priors.p_init),
    )


def _prior_from_config(cfg: DictConfig) -> Prior:
    """Build a `Prior` from a Hydra config node."""
    kwargs = OmegaConf.to_container(cfg.get("kwargs", {}), resolve=True)
    return Prior(cfg.dist, dict(kwargs))


def _shape_group(num_transmitted_alleles: int) -> str:
    return f"num_transmitted_alleles={int(num_transmitted_alleles)}"


def _num_hosts_dtype(num_hosts: int) -> np.dtype:
    if num_hosts <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _validate_num_hosts_by_state(
    num_hosts_by_state: np.ndarray,
    expected_num_hosts: np.ndarray,
    context: str = "",
) -> None:
    observed_num_hosts = num_hosts_by_state.sum(axis=-1)
    expected = expected_num_hosts[:, None]
    if np.all(observed_num_hosts == expected):
        return
    observed_min = int(np.min(observed_num_hosts))
    observed_max = int(np.max(observed_num_hosts))
    expected_min = int(np.min(expected_num_hosts))
    expected_max = int(np.max(expected_num_hosts))
    suffix = f" {context}" if context else ""
    raise ValueError(
        "Host histogram counts do not match per-simulation num_hosts"
        f"{suffix}: observed [{observed_min}, {observed_max}], "
        f"expected [{expected_min}, {expected_max}]."
    )


def make_histogram_dataset(
    batch: MalariaSimulationBatch,
    sample_ids: np.ndarray,
    generation_ids: np.ndarray,
    seed: int,
    context: str = "",
) -> xr.Dataset:
    parameters = np.asarray(jax.device_get(batch.parameters), dtype=np.float32)
    expected_num_hosts = parameters[..., 0].astype(np.int64, copy=False)
    max_num_hosts = int(expected_num_hosts.max())
    num_hosts_by_state = np.asarray(jax.device_get(batch.num_hosts_by_state)).astype(
        _num_hosts_dtype(max_num_hosts),
        copy=False,
    )
    _validate_num_hosts_by_state(
        num_hosts_by_state,
        expected_num_hosts,
        context=context,
    )
    state_num_mutants = np.asarray(
        jax.device_get(batch.state_num_mutants),
        dtype=np.uint16,
    )
    state_num_alleles = np.asarray(
        jax.device_get(batch.state_num_alleles),
        dtype=np.uint16,
    )

    return xr.Dataset(
        data_vars={
            "num_hosts_by_state": (
                ("simulation", "generation", "state"),
                num_hosts_by_state,
            ),
            "parameters": (
                ("simulation", "parameter"),
                parameters,
            ),
        },
        coords={
            "simulation": sample_ids.astype(np.int64, copy=False),
            "generation": generation_ids.astype(np.int32, copy=False),
            "state": np.arange(state_num_mutants.size, dtype=np.int16),
            "state_num_mutants": ("state", state_num_mutants),
            "state_num_alleles": ("state", state_num_alleles),
            "state_allele_freq": (
                "state",
                state_num_mutants.astype(np.float32)
                / state_num_alleles.astype(np.float32),
            ),
            "parameter": np.asarray(batch.parameter_names, dtype=object),
        },
        attrs={
            "simulator": "malaria",
            "num_transmitted_alleles": int(batch.num_transmitted_alleles),
            "batch_p_init": float(batch.batch_p_init),
            "max_num_hosts": max_num_hosts,
            "seed": int(seed),
            "parameter_names": list(batch.parameter_names),
        },
    )


def _flush_datasets(
    datasets: list[xr.Dataset],
    store: Path,
    group: str,
    group_num_samples: dict[str, int],
    output_cfg: DictConfig,
) -> None:
    if not datasets:
        return
    ds = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="simulation")
    append = group_num_samples.get(group, 0) > 0
    _write_dataset(ds, store, group, append=append, output_cfg=output_cfg)
    group_num_samples[group] = group_num_samples.get(group, 0) + ds.sizes["simulation"]


def _write_dataset(
    ds: xr.Dataset,
    store: Path,
    group: str,
    append: bool,
    output_cfg: DictConfig,
) -> None:
    chunks = output_cfg.chunks
    num_simulations_chunk = min(
        int(chunks.num_simulations),
        ds.sizes["simulation"],
    )
    num_generations_chunk = min(
        int(chunks.num_generations),
        ds.sizes["generation"],
    )
    num_states_chunk = min(int(chunks.num_states), ds.sizes["state"])
    encoding = {
        "num_hosts_by_state": {
            "chunks": (
                num_simulations_chunk,
                num_generations_chunk,
                num_states_chunk,
            ),
        },
        "parameters": {"chunks": (num_simulations_chunk, ds.sizes["parameter"])},
    }
    compressors = _compressors(output_cfg)
    if compressors is not None:
        for variable_encoding in encoding.values():
            variable_encoding["compressors"] = compressors
    kwargs = {
        "store": store,
        "group": group,
        "mode": "a",
        "consolidated": False,
        "write_empty_chunks": bool(output_cfg.write_empty_chunks),
    }
    if append:
        kwargs["append_dim"] = "simulation"
    else:
        kwargs["encoding"] = encoding
    ds.to_zarr(**kwargs)


def _compressors(output_cfg: DictConfig) -> list[BloscCodec] | None:
    compression = output_cfg.compression
    if not compression.enabled:
        return None
    if compression.codec != "blosc":
        raise ValueError(f"Unsupported compression codec: {compression.codec}")
    return [
        BloscCodec(
            cname=compression.cname,
            clevel=int(compression.clevel),
            shuffle=compression.shuffle,
        )
    ]


def _write_root_attrs(
    store: Path,
    cfg: DictConfig,
    groups: list[dict],
) -> None:
    root = zarr.open_group(str(store), mode="a")
    root.attrs.update(
        {
            "simulator": "malaria",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "groups": groups,
        }
    )


if __name__ == "__main__":
    main()
