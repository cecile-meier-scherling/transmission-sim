import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Malaria Selection Trajectory Analysis

    This notebook explores:
    - trajectory dynamics of **mean allele frequency** and **prevalence**
      under separate 1D sweeps of `sw` and `st`;
    - a joint `sw x st` heatmap of final-generation outcomes;
    - the approximation
      \[
      \mathbb{E}[M_{t+1} \mid M_t=1] \approx
      \frac{Q(1+s_w)}{Q+s_w}
      \left(1+\frac{s_t}{Q}\right)
      (1+\chi)
      \]
      compared with the exact one-step expectation from the simulator kernel.

    Note: generation `0` is the founding distribution and depends on `p_init`.
    Selection effects begin at generation `1`.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from jax import random

    from dl4bi_sps.malaria import MalariaTransmission
    from dl4bi_sps.priors import Prior

    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from malaria_analysis import (
        exact_lineage_expectation,
        initial_state_m_equals_one,
        lineage_expectation_formula,
        run_trajectory,
    )

    return (
        MalariaTransmission,
        Prior,
        exact_lineage_expectation,
        initial_state_m_equals_one,
        lineage_expectation_formula,
        np,
        plt,
        random,
        run_trajectory,
    )


@app.cell
def _(mo):
    num_hosts = mo.ui.number(start=64, stop=8192, step=64, value=1024, label="num_hosts")
    q = mo.ui.number(start=1, stop=16, step=1, value=4, label="Q (num_transmitted_alleles)")
    chi = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.25, label="chi")
    p_init = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.05, label="p_init")
    num_generations = mo.ui.number(
        start=2, stop=512, step=1, value=64, label="num_generations"
    )
    num_simulations = mo.ui.number(
        start=8, stop=1024, step=8, value=128, label="num_simulations"
    )
    seed = mo.ui.number(start=0, stop=1_000_000, step=1, value=42, label="seed")

    fixed_st = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.4, label="fixed st")
    sw_min = mo.ui.slider(start=-0.999, stop=0.0, step=0.01, value=-0.9, label="sw min")
    sw_max = mo.ui.slider(start=-0.999, stop=0.0, step=0.01, value=0.0, label="sw max")
    sw_num = mo.ui.number(start=2, stop=40, step=1, value=10, label="sw points")

    fixed_sw = mo.ui.slider(start=-0.999, stop=0.0, step=0.01, value=-0.2, label="fixed sw")
    st_min = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.0, label="st min")
    st_max = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=1.0, label="st max")
    st_num = mo.ui.number(start=2, stop=40, step=1, value=10, label="st points")

    lineage_sw = mo.ui.slider(
        start=-0.999, stop=0.0, step=0.01, value=-0.2, label="lineage fixed sw"
    )
    lineage_st = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.4, label="lineage fixed st"
    )
    lineage_num = mo.ui.number(
        start=5, stop=80, step=1, value=25, label="lineage sweep points"
    )

    joint_sw_min = mo.ui.slider(
        start=-0.999, stop=0.0, step=0.01, value=-0.9, label="joint sw min"
    )
    joint_sw_max = mo.ui.slider(
        start=-0.999, stop=0.0, step=0.01, value=0.0, label="joint sw max"
    )
    joint_sw_points = mo.ui.number(
        start=2, stop=30, step=1, value=8, label="joint sw points"
    )
    joint_st_min = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.0, label="joint st min")
    joint_st_max = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=1.0, label="joint st max")
    joint_st_points = mo.ui.number(
        start=2, stop=30, step=1, value=8, label="joint st points"
    )

    controls = mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([num_hosts, q, chi, p_init]),
            mo.hstack([num_generations, num_simulations, seed]),
            mo.md("### Sweep ranges"),
            mo.hstack([fixed_st, sw_min, sw_max, sw_num]),
            mo.hstack([fixed_sw, st_min, st_max, st_num]),
            mo.md("### Joint sw x st grid"),
            mo.hstack(
                [
                    joint_sw_min,
                    joint_sw_max,
                    joint_sw_points,
                    joint_st_min,
                    joint_st_max,
                    joint_st_points,
                ]
            ),
            mo.md("### Lineage panel"),
            mo.hstack([lineage_sw, lineage_st, lineage_num]),
        ]
    )
    controls
    return (
        chi,
        fixed_st,
        fixed_sw,
        joint_st_max,
        joint_st_min,
        joint_st_points,
        joint_sw_max,
        joint_sw_min,
        joint_sw_points,
        lineage_num,
        lineage_st,
        lineage_sw,
        num_generations,
        num_hosts,
        num_simulations,
        p_init,
        q,
        seed,
        st_max,
        st_min,
        st_num,
        sw_max,
        sw_min,
        sw_num,
    )


@app.cell
def _(
    MalariaTransmission,
    Prior,
    chi,
    fixed_st,
    np,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    q,
    random,
    run_trajectory,
    seed,
    sw_max,
    sw_min,
    sw_num,
):
    def compute_sw_sweep():
        sw_values = np.linspace(sw_min.value, sw_max.value, int(sw_num.value), dtype=float)
        rng_sw = random.key(int(seed.value))
        mean_by_sw = []
        prev_by_sw = []
        for sw_i in sw_values:
            rng_sw, rng_sw_i = random.split(rng_sw)
            sim_sw = MalariaTransmission(
                num_hosts=Prior("fixed", {"value": int(num_hosts.value)}),
                chi=Prior("fixed", {"value": float(chi.value)}),
                sw=Prior("fixed", {"value": float(sw_i)}),
                st=Prior("fixed", {"value": float(fixed_st.value)}),
            )
            mean_traj_sw, prev_traj_sw = run_trajectory(
                sim=sim_sw,
                rng=rng_sw_i,
                num_generations=int(num_generations.value),
                num_simulations=int(num_simulations.value),
                num_transmitted_alleles=int(q.value),
                p_init=float(p_init.value),
            )
            mean_by_sw.append(np.asarray(mean_traj_sw))
            prev_by_sw.append(np.asarray(prev_traj_sw))
        return mean_by_sw, prev_by_sw, sw_values

    mean_by_sw, prev_by_sw, sw_values = compute_sw_sweep()
    return mean_by_sw, prev_by_sw, sw_values


@app.cell
def _(
    chi,
    fixed_st,
    mean_by_sw,
    np,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    plt,
    prev_by_sw,
    q,
    sw_values,
):
    def render_sw_panel():
        sw_generations = np.arange(int(num_generations.value))
        sw_fig, sw_axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
        sw_cmap = plt.get_cmap("viridis")
        for sw_idx, sw_i in enumerate(sw_values):
            sw_color = sw_cmap(sw_idx / max(len(sw_values) - 1, 1))
            mean_traj_sw = mean_by_sw[sw_idx]
            prev_traj_sw = prev_by_sw[sw_idx]
            mean_mu_sw = mean_traj_sw.mean(axis=0)
            mean_sd_sw = mean_traj_sw.std(axis=0)
            prev_mu_sw = prev_traj_sw.mean(axis=0)
            prev_sd_sw = prev_traj_sw.std(axis=0)
            sw_axes[0].plot(
                sw_generations,
                mean_mu_sw,
                color=sw_color,
                lw=1.4,
                label=f"sw={sw_i:.2f}",
            )
            sw_axes[0].fill_between(
                sw_generations,
                mean_mu_sw - mean_sd_sw,
                mean_mu_sw + mean_sd_sw,
                color=sw_color,
                alpha=0.15,
            )
            sw_axes[1].plot(sw_generations, prev_mu_sw, color=sw_color, lw=1.4)
            sw_axes[1].fill_between(
                sw_generations,
                prev_mu_sw - prev_sd_sw,
                prev_mu_sw + prev_sd_sw,
                color=sw_color,
                alpha=0.15,
            )
        sw_axes[0].set_title("Mean allele frequency vs generation (sw sweep)")
        sw_axes[1].set_title("Prevalence vs generation (sw sweep)")
        sw_axes[0].set_xlabel("generation")
        sw_axes[1].set_xlabel("generation")
        sw_axes[0].set_ylabel("mean allele frequency")
        sw_axes[1].set_ylabel("prevalence")
        sw_axes[0].legend(loc="best", fontsize=8, ncol=2)
        sw_fixed_text = (
            f"fixed: st={fixed_st.value:.2f}, chi={chi.value:.2f}, "
            f"p_init={p_init.value:.2f}, Q={int(q.value)}, "
            f"num_hosts={int(num_hosts.value)}, "
            f"num_simulations={int(num_simulations.value)}"
        )
        sw_fig.text(0.5, 0.01, sw_fixed_text, ha="center", va="bottom", fontsize=9)
        sw_fig.subplots_adjust(bottom=0.18)
        return sw_fig

    sw_fig = render_sw_panel()
    sw_fig
    return


@app.cell
def _(
    MalariaTransmission,
    Prior,
    chi,
    fixed_sw,
    np,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    q,
    random,
    run_trajectory,
    seed,
    st_max,
    st_min,
    st_num,
):
    def compute_st_sweep():
        st_values = np.linspace(st_min.value, st_max.value, int(st_num.value), dtype=float)
        rng_st = random.key(int(seed.value) + 1)
        mean_by_st = []
        prev_by_st = []
        for st_i in st_values:
            rng_st, rng_st_i = random.split(rng_st)
            sim_st = MalariaTransmission(
                num_hosts=Prior("fixed", {"value": int(num_hosts.value)}),
                chi=Prior("fixed", {"value": float(chi.value)}),
                sw=Prior("fixed", {"value": float(fixed_sw.value)}),
                st=Prior("fixed", {"value": float(st_i)}),
            )
            mean_traj_st, prev_traj_st = run_trajectory(
                sim=sim_st,
                rng=rng_st_i,
                num_generations=int(num_generations.value),
                num_simulations=int(num_simulations.value),
                num_transmitted_alleles=int(q.value),
                p_init=float(p_init.value),
            )
            mean_by_st.append(np.asarray(mean_traj_st))
            prev_by_st.append(np.asarray(prev_traj_st))
        return mean_by_st, prev_by_st, st_values

    mean_by_st, prev_by_st, st_values = compute_st_sweep()
    return mean_by_st, prev_by_st, st_values


@app.cell
def _(
    chi,
    fixed_sw,
    mean_by_st,
    np,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    plt,
    prev_by_st,
    q,
    st_values,
):
    def render_st_panel():
        st_generations = np.arange(int(num_generations.value))
        st_fig, st_axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
        st_cmap = plt.get_cmap("plasma")
        for st_idx, st_i in enumerate(st_values):
            st_color = st_cmap(st_idx / max(len(st_values) - 1, 1))
            mean_traj_st = mean_by_st[st_idx]
            prev_traj_st = prev_by_st[st_idx]
            mean_mu_st = mean_traj_st.mean(axis=0)
            mean_sd_st = mean_traj_st.std(axis=0)
            prev_mu_st = prev_traj_st.mean(axis=0)
            prev_sd_st = prev_traj_st.std(axis=0)
            st_axes[0].plot(
                st_generations,
                mean_mu_st,
                color=st_color,
                lw=1.4,
                label=f"st={st_i:.2f}",
            )
            st_axes[0].fill_between(
                st_generations,
                mean_mu_st - mean_sd_st,
                mean_mu_st + mean_sd_st,
                color=st_color,
                alpha=0.15,
            )
            st_axes[1].plot(st_generations, prev_mu_st, color=st_color, lw=1.4)
            st_axes[1].fill_between(
                st_generations,
                prev_mu_st - prev_sd_st,
                prev_mu_st + prev_sd_st,
                color=st_color,
                alpha=0.15,
            )
        st_axes[0].set_title("Mean allele frequency vs generation (st sweep)")
        st_axes[1].set_title("Prevalence vs generation (st sweep)")
        st_axes[0].set_xlabel("generation")
        st_axes[1].set_xlabel("generation")
        st_axes[0].set_ylabel("mean allele frequency")
        st_axes[1].set_ylabel("prevalence")
        st_axes[0].legend(loc="best", fontsize=8, ncol=2)
        st_fixed_text = (
            f"fixed: sw={fixed_sw.value:.2f}, chi={chi.value:.2f}, "
            f"p_init={p_init.value:.2f}, Q={int(q.value)}, "
            f"num_hosts={int(num_hosts.value)}, "
            f"num_simulations={int(num_simulations.value)}"
        )
        st_fig.text(0.5, 0.01, st_fixed_text, ha="center", va="bottom", fontsize=9)
        st_fig.subplots_adjust(bottom=0.18)
        return st_fig

    st_fig = render_st_panel()
    st_fig
    return


@app.cell
def _(
    MalariaTransmission,
    Prior,
    chi,
    joint_st_max,
    joint_st_min,
    joint_st_points,
    joint_sw_max,
    joint_sw_min,
    joint_sw_points,
    np,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    q,
    random,
    run_trajectory,
    seed,
):
    def compute_joint_grid():
        joint_sw_values = np.linspace(
            joint_sw_min.value,
            joint_sw_max.value,
            int(joint_sw_points.value),
            dtype=float,
        )
        joint_st_values = np.linspace(
            joint_st_min.value,
            joint_st_max.value,
            int(joint_st_points.value),
            dtype=float,
        )
        rng_joint = random.key(int(seed.value) + 2)
        final_mean_grid = np.zeros((joint_sw_values.size, joint_st_values.size), dtype=float)
        final_prev_grid = np.zeros((joint_sw_values.size, joint_st_values.size), dtype=float)
        for joint_sw_idx, joint_sw_i in enumerate(joint_sw_values):
            for joint_st_idx, joint_st_i in enumerate(joint_st_values):
                rng_joint, rng_joint_i = random.split(rng_joint)
                sim_joint = MalariaTransmission(
                    num_hosts=Prior("fixed", {"value": int(num_hosts.value)}),
                    chi=Prior("fixed", {"value": float(chi.value)}),
                    sw=Prior("fixed", {"value": float(joint_sw_i)}),
                    st=Prior("fixed", {"value": float(joint_st_i)}),
                )
                mean_traj_joint, prev_traj_joint = run_trajectory(
                    sim=sim_joint,
                    rng=rng_joint_i,
                    num_generations=int(num_generations.value),
                    num_simulations=int(num_simulations.value),
                    num_transmitted_alleles=int(q.value),
                    p_init=float(p_init.value),
                )
                final_gen = int(num_generations.value) - 1
                final_mean_grid[joint_sw_idx, joint_st_idx] = float(
                    np.asarray(mean_traj_joint)[:, final_gen].mean()
                )
                final_prev_grid[joint_sw_idx, joint_st_idx] = float(
                    np.asarray(prev_traj_joint)[:, final_gen].mean()
                )
        return final_mean_grid, final_prev_grid, joint_sw_values, joint_st_values

    final_mean_grid, final_prev_grid, joint_sw_values, joint_st_values = (
        compute_joint_grid()
    )
    return (
        final_mean_grid,
        final_prev_grid,
        joint_st_values,
        joint_sw_values,
    )


@app.cell
def _(
    chi,
    final_mean_grid,
    final_prev_grid,
    joint_st_values,
    joint_sw_values,
    num_generations,
    num_hosts,
    num_simulations,
    p_init,
    plt,
    q,
):
    def render_joint_heatmaps():
        joint_fig, joint_axes = plt.subplots(1, 2, figsize=(13, 5.2))
        joint_extent = [
            float(joint_st_values[0]),
            float(joint_st_values[-1]),
            float(joint_sw_values[0]),
            float(joint_sw_values[-1]),
        ]
        joint_mean_plot = joint_axes[0].imshow(
            final_mean_grid,
            origin="lower",
            aspect="auto",
            extent=joint_extent,
            cmap="viridis",
        )
        joint_prev_plot = joint_axes[1].imshow(
            final_prev_grid,
            origin="lower",
            aspect="auto",
            extent=joint_extent,
            cmap="plasma",
        )
        joint_axes[0].set_title(
            f"Final mean allele frequency (gen {int(num_generations.value) - 1})"
        )
        joint_axes[1].set_title(
            f"Final prevalence (gen {int(num_generations.value) - 1})"
        )
        joint_axes[0].set_xlabel("st")
        joint_axes[1].set_xlabel("st")
        joint_axes[0].set_ylabel("sw")
        joint_axes[1].set_ylabel("sw")
        joint_fig.colorbar(joint_mean_plot, ax=joint_axes[0], fraction=0.046, pad=0.04)
        joint_fig.colorbar(joint_prev_plot, ax=joint_axes[1], fraction=0.046, pad=0.04)
        joint_fixed_text = (
            f"fixed: chi={chi.value:.2f}, p_init={p_init.value:.2f}, "
            f"Q={int(q.value)}, num_hosts={int(num_hosts.value)}, "
            f"num_simulations={int(num_simulations.value)}"
        )
        joint_fig.text(0.5, 0.01, joint_fixed_text, ha="center", va="bottom", fontsize=9)
        joint_fig.subplots_adjust(bottom=0.16)
        return joint_fig

    joint_fig = render_joint_heatmaps()
    joint_fig
    return


@app.cell
def _(
    chi,
    exact_lineage_expectation,
    initial_state_m_equals_one,
    lineage_expectation_formula,
    lineage_num,
    lineage_st,
    lineage_sw,
    np,
    num_hosts,
    plt,
    q,
):
    def render_lineage_panel():
        lineage_sw_values = np.linspace(-0.95, 0.0, int(lineage_num.value), dtype=float)
        lineage_st_values = np.linspace(0.0, 1.0, int(lineage_num.value), dtype=float)
        chi_values = np.linspace(0.0, 1.0, int(lineage_num.value), dtype=float)

        exact_vs_sw = []
        formula_vs_sw = []
        for lineage_sw_i in lineage_sw_values:
            state_sw = initial_state_m_equals_one(
                chi=float(chi.value),
                sw=float(lineage_sw_i),
                st=float(lineage_st.value),
                num_hosts=int(num_hosts.value),
                Q=int(q.value),
            )
            exact_vs_sw.append(float(exact_lineage_expectation(state_sw)[0]))
            formula_vs_sw.append(
                float(
                    lineage_expectation_formula(
                        Q=int(q.value),
                        sw=lineage_sw_i,
                        st=float(lineage_st.value),
                        chi=float(chi.value),
                    )
                )
            )

        exact_vs_st = []
        formula_vs_st = []
        for lineage_st_i in lineage_st_values:
            state_st = initial_state_m_equals_one(
                chi=float(chi.value),
                sw=float(lineage_sw.value),
                st=float(lineage_st_i),
                num_hosts=int(num_hosts.value),
                Q=int(q.value),
            )
            exact_vs_st.append(float(exact_lineage_expectation(state_st)[0]))
            formula_vs_st.append(
                float(
                    lineage_expectation_formula(
                        Q=int(q.value),
                        sw=float(lineage_sw.value),
                        st=lineage_st_i,
                        chi=float(chi.value),
                    )
                )
            )

        exact_vs_chi = []
        formula_vs_chi = []
        for chi_i in chi_values:
            state_chi = initial_state_m_equals_one(
                chi=float(chi_i),
                sw=float(lineage_sw.value),
                st=float(lineage_st.value),
                num_hosts=int(num_hosts.value),
                Q=int(q.value),
            )
            exact_vs_chi.append(float(exact_lineage_expectation(state_chi)[0]))
            formula_vs_chi.append(
                float(
                    lineage_expectation_formula(
                        Q=int(q.value),
                        sw=float(lineage_sw.value),
                        st=float(lineage_st.value),
                        chi=chi_i,
                    )
                )
            )
        rel_err_chi = (np.asarray(formula_vs_chi) - np.asarray(exact_vs_chi)) / np.maximum(
            np.asarray(exact_vs_chi), 1e-12
        )

        lineage_fig, lineage_axes = plt.subplots(1, 3, figsize=(16, 4))
        lineage_axes[0].plot(lineage_sw_values, exact_vs_sw, label="exact", lw=2)
        lineage_axes[0].plot(
            lineage_sw_values,
            formula_vs_sw,
            label="formula",
            lw=2,
            ls="--",
        )
        lineage_axes[0].set_title("E[M(t+1)|M(t)=1] vs sw")
        lineage_axes[0].set_xlabel("sw")
        lineage_axes[0].set_ylabel("expected mutant lineages")
        lineage_axes[0].legend(loc="best")

        lineage_axes[1].plot(lineage_st_values, exact_vs_st, label="exact", lw=2)
        lineage_axes[1].plot(
            lineage_st_values,
            formula_vs_st,
            label="formula",
            lw=2,
            ls="--",
        )
        lineage_axes[1].set_title("E[M(t+1)|M(t)=1] vs st")
        lineage_axes[1].set_xlabel("st")
        lineage_axes[1].set_ylabel("expected mutant lineages")
        lineage_axes[1].legend(loc="best")

        lineage_axes[2].plot(chi_values, rel_err_chi, lw=2)
        lineage_axes[2].axhline(0.0, color="black", lw=1, ls=":")
        lineage_axes[2].set_title("Relative formula error vs chi")
        lineage_axes[2].set_xlabel("chi")
        lineage_axes[2].set_ylabel("(formula - exact) / exact")
        lineage_fig.tight_layout()
        return lineage_fig

    lineage_fig = render_lineage_panel()
    lineage_fig
    return


if __name__ == "__main__":
    app.run()
