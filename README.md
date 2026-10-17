# Stochastic Process Simulators (`dl4bi-sps`)

## Overview

### Simulator

This model extends the [parasite transmission method published in Wellcome Open
Research](https://wellcomeopenresearch.org/articles/9-215) by incorporating
within-host ($s_w$) and transmission ($s_t$) selection coefficients.

The malaria simulator (`dl4bi_sps.malaria.MalariaTransmission`) models parasite
transmission across discrete generations. Conceptually, transmission forms a
directed acyclic graph in which nodes are human hosts and edges are
mosquito-mediated transmission events. The implementation tracks counts of host
states (mutant and total allele counts), rather than storing individual hosts or
edges. Each generation contains a fixed effective population of $N_h$ hosts
from which transmission sources are sampled.

Selection acts at two distinct stages:

- **Within-host selection** ($s_w \in [-1, 0)$): represents the replicative
  fitness cost of K13 resistance mutations in the absence of drug pressure.
  Mutant alleles receive weight $1 + s_w$ during within-host resampling, while
  wild-type alleles receive weight 1. Setting `sw=0` disables this resampling.
- **Transmission selection** ($s_t \in [0, 1]$): represents the survival
  advantage of hosts carrying K13 mutant alleles under artemisinin treatment.
  A host with mutant allele frequency $p$ receives source-selection weight
  $1 + s_t p$. Treatment is represented by this weight, rather than simulated
  explicitly.

Each vector bite transmits $Q$ alleles. With probability `chi`, a recipient
is superinfected and receives contributions from two distinct source hosts,
for $2Q$ alleles in total. Within-host selection acts on the resulting allele
pool. Multinomial sampling of the next
generation preserves the host population size and introduces genetic drift.
Founding hosts carry $Q$ alleles, with mutant counts drawn using `p_init`.

See [dataset generation](examples/build_malaria_dataset.py), its
[configuration](examples/configs/malaria_dataset.yaml), and the
[selection analysis](examples/analyze_malaria_selection.py).

## Model Parameters

`Nh` and `Q` correspond to `num_hosts` and `num_transmitted_alleles` in the Python
API. Simulator configuration uses `Prior` objects to specify fixed values or
sampling distributions.

| Parameter | Python name | Type | Description |
|-----------|-------------|------|-------------|
| `Nh` | `num_hosts` | `int` | Effective number of hosts per generation; clamped to at least 2 |
| `Q` | `num_transmitted_alleles` | `int` | Number of alleles transmitted per vector bite; at least 1 |
| `chi` | `chi` | `float` | Superinfection probability, between 0 and 1 |
| `sw` | `sw` | `float` | Within-host selection coefficient, $s_w \in [-1, 0)$; 0 is also supported as a neutral control |
| `st` | `st` | `float` | Transmission selection coefficient, $s_t \in [0, 1]$ |
| `p_init` | `p_init` | `float` | Initial K13 mutant allele frequency, between 0 and 1 |

Mutation is not modeled in the current implementation; `mu` is not an active
parameter.

Within each simulation batch, `Q` and `p_init` are shared, while `Nh`,
`chi`, `sw`, and `st` are sampled independently for each simulation. Output
host-state counts have shape `(num_simulations, num_generations, 3 * Q + 2)`;
the first generation is the founding population.

## Installation

Requirements:

- Python **3.11 or newer, below 4.0**, as declared in `pyproject.toml`.
- [uv](https://github.com/astral-sh/uv) for dependency management, and Git to
  clone the source repository.
- A CPU for the default setup, or an NVIDIA GPU with a compatible driver for
  the optional CUDA 12 or CUDA 13 JAX backend.

`uv sync` installs the project and its dependencies, including JAX, Matplotlib,
Hydra/OmegaConf, tqdm, Tweepy, Xarray, and Zarr. The lockfile records resolved
versions; development dependencies include pytest, pytest-benchmark, and marimo.

```bash
git clone https://github.com/cecile-meier-scherling/transmission-sim.git
cd transmission-sim
uv sync --extra cpu
```

For GPU execution, replace `--extra cpu` with `--extra cuda12` or
`--extra cuda13`. Select exactly one backend extra; these extras conflict with
each other. A plain `uv sync` also installs the base JAX dependency.

Alternatively, install the published package with pip. The distribution name is
`dl4bi-sps`, and the import package is `dl4bi_sps`. Choose one command:

```bash
pip install dl4bi-sps
pip install 'dl4bi-sps[cpu]'
pip install 'dl4bi-sps[cuda12]'
pip install 'dl4bi-sps[cuda13]'
```

## View Documentation (Locally)

After installing from source:

```bash
uv run --with pdoc pdoc --docformat google --math dl4bi_sps
```

## Demo
```python
import matplotlib.pyplot as plt

import jax
from jax import random

from dl4bi_sps.gp import GP
from dl4bi_sps.priors import Prior
from dl4bi_sps.utils import build_grid
from dl4bi_sps.kernels import matern_3_2, matern_5_2

rng = random.key(42)

s_1d = build_grid([{"start": -2, "stop": 2, "num": 128}])
s_2d = build_grid([{"start": -1.5, "stop": 1.5, "num": 300}, {"start": -2.5, "stop": 2.5, "num": 500}])
batch_size = 1
approx = True
lengthscales = [0.05, 0.1, 0.2]
for name, s in zip(["1d", "2d"], [s_1d, s_2d]):
    fig, axes = plt.subplots(len(lengthscales), 1)
    for i, ls in enumerate(lengthscales):
        gp = GP(matern_3_2, ls=Prior("fixed", {"value": ls}))
        f, *_ = gp.simulate(rng, s, batch_size, approx)
        axes[i].set_title(f"ls={ls}")
        if name == "1d":
            axes[i].plot(s, f.squeeze().T)
        else:
            axes[i].imshow(f.squeeze().reshape(300, 500), cmap="Spectral_r")
    plt.tight_layout()
    plt.savefig(f"{name}_gp.png", dpi=150)
    plt.clf()

# create a simple (forever) dataloader
def dataloader(rng, gp, s, batch_size=64, approx=False):
    while True:
        rng_i, rng = random.split(rng)
        yield gp.simulate(rng_i, s, batch_size, approx)


gp = GP(matern_5_2, ls=Prior("beta", {"a": 2.5, "b": 5}))
loader = dataloader(rng, gp, s, batch_size, approx=True)
f, var, ls, period, z = next(loader)


# within IPython, speed test Kronecker (approx) vs. Cholesky methods 
rng, batch_size = random.key(42), 1024
s = build_grid([{"start": 0, "stop": 1, "num": 64}] * 2) # 64x64 grid
%timeit gp.simulate(rng, s, batch_size, approx=True) # ~5 ms
%timeit gp.simulate(rng, s, batch_size, approx=False) # ~50 ms
````
More examples can be found [here](https://github.com/MLGlobalHealth/sps/tree/main/examples).

## Gotchas
Small lengthscales can cause numerical instability. Enabling 64-bit floating
operations often helps, but it roughly doubles memory usage and may reduce
throughput on accelerators.
```python
import jax
# use 64-bit precision globally
jax.config.update("jax_enable_x64", True)
# use 64-bit precision only inside this context manager
with jax.enable_x64():
    # Do something in 64-bit precision
    ...
# Back to default 32-bit precision
```

## Development Setup
- Install [uv](https://docs.astral.sh/uv/).
- Clone the repository and enter `transmission-sim` using the commands above.
- Install a compatible Python version if needed: `uv python install 3.11`
- Sync the project, development dependencies, and one JAX extra: `uv sync --extra cpu` (or choose `cuda12` or `cuda13`)
- Run the test suite: `uv run pytest`

`uv sync` creates a local `.venv/` and installs the project in editable mode,
so changes in `dl4bi_sps/` are reflected immediately.

## Build and Publish to PyPI
Create a local `.env` file with the publish tokens:
```env
TEST_PYPI_TOKEN=pypi-...
PYPI_TOKEN=pypi-...
```

Run the release helper from a clean `main` checkout:
```bash
uv run python scripts/release.py .env "Release notes"
```

The helper bumps the patch version, commits and tags `v<version> <message>`,
rebuilds `dist/`, publishes to TestPyPI and PyPI, smoke-tests the published
install targets for `dl4bi-sps`, then pushes `main` and the tag.

To rerun only the install smoke tests for an already-published version:
```bash
uv run python scripts/release.py --smoke-only 0.1.2
```

To limit the smoke-only run to specific targets:
```bash
uv run python scripts/release.py --smoke-only 0.1.2 --target base --target cpu
```
