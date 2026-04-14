#!/usr/bin/env python
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vesin import NeighborList


ALGORITHMS = ["auto", "brute_force", "cell_list"]
ALGORITHM_LABELS = {
    "auto": "Auto",
    "brute_force": "Brute Force",
    "cell_list": "Cell List",
}


def generate_system(n_atoms, density, seed=42):
    box_size = (n_atoms / density) ** (1 / 3)
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * box_size
    box = np.eye(3) * box_size
    return positions, box


def benchmark_cuda(
    positions_gpu, box_gpu, cutoff, algorithm="auto", n_warmup=10, n_runs=25
):
    nl = NeighborList(cutoff=cutoff, full_list=True, algorithm=algorithm)

    # Warmup
    for _ in range(n_warmup):
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
    cp.cuda.Stream.null.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    times = []

    for _ in range(n_runs):
        start_event.record()
        nl.compute(positions_gpu, box_gpu, periodic=True, quantities="ij")
        end_event.record()
        end_event.synchronize()
        times.append(cp.cuda.get_elapsed_time(start_event, end_event))

    return np.mean(times), np.std(times)


def run_benchmarks(
    n_atoms_list,
    cutoffs,
    min_particles_values,
    algorithms=None,
    density=1.0,
    n_warmup=10,
    n_runs=25,
):
    if algorithms is None:
        algorithms = ALGORITHMS

    # results[algorithm][cutoff][n_atoms][min_particles] = (mean, std)
    results = {algo: {} for algo in algorithms}

    for algorithm in algorithms:
        print(f"\n{'#' * 60}")
        print(f"ALGORITHM = {ALGORITHM_LABELS[algorithm]}")
        print("#" * 60)

        for cutoff in cutoffs:
            print(f"\n{'=' * 60}")
            print(f"CUTOFF = {cutoff}")
            print("=" * 60)

            results[algorithm][cutoff] = {}

            for n_atoms in n_atoms_list:
                positions, box = generate_system(n_atoms, density)
                box_size = box[0, 0]

                positions_gpu = cp.asarray(positions)
                box_gpu = cp.asarray(box)

                print(f"\n{n_atoms} atoms (box={box_size:.1f}):")
                results[algorithm][cutoff][n_atoms] = {}

                for min_particles in min_particles_values:
                    # Set environment variable
                    os.environ["VESIN_CUDA_MIN_PARTICLES_PER_CELL"] = str(min_particles)

                    try:
                        mean, std = benchmark_cuda(
                            positions_gpu,
                            box_gpu,
                            cutoff,
                            algorithm=algorithm,
                            n_warmup=n_warmup,
                            n_runs=n_runs,
                        )
                        results[algorithm][cutoff][n_atoms][min_particles] = (
                            mean,
                            std,
                        )
                        print(
                            f"   MIN_PARTICLES_PER_CELL={min_particles:4d}: "
                            f"{mean:8.3f} ± {std:.3f} ms"
                        )
                    except Exception as e:
                        print(
                            f"   MIN_PARTICLES_PER_CELL={min_particles:4d}: ERROR ({e})"
                        )
                        results[algorithm][cutoff][n_atoms][min_particles] = (
                            np.nan,
                            np.nan,
                        )

    if "VESIN_CUDA_MIN_PARTICLES_PER_CELL" in os.environ:
        del os.environ["VESIN_CUDA_MIN_PARTICLES_PER_CELL"]

    return results


def print_summary(results, min_particles_values):
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for algorithm, algo_results in results.items():
        print(f"\nAlgorithm: {ALGORITHM_LABELS[algorithm]}")
        print("=" * 80)

        for cutoff, cutoff_results in algo_results.items():
            print(f"\n  Cutoff = {cutoff}")
            print("  " + "-" * 78)

            # Header
            header = f"  {'n_atoms':>10}"
            for mp in min_particles_values:
                header += f" | {mp:>10}"
            print(header)
            print("  " + "-" * 78)

            for n_atoms, atom_results in cutoff_results.items():
                row = f"  {n_atoms:>10}"
                for mp in min_particles_values:
                    if mp in atom_results:
                        mean, _ = atom_results[mp]
                        if np.isnan(mean):
                            row += f" | {'ERROR':>10}"
                        else:
                            row += f" | {mean:>10.3f}"
                    else:
                        row += f" | {'-':>10}"
                print(row)


def plot_results(
    results,
    min_particles_values,
    output_file="min_particles_per_cell_benchmark.png",
):
    sns.set_theme(style="whitegrid")

    algorithms = list(results.keys())
    # Get cutoffs from the first algorithm's results
    cutoffs = list(results[algorithms[0]].keys())
    n_algorithms = len(algorithms)
    n_cutoffs = len(cutoffs)

    fig, axes = plt.subplots(
        n_algorithms,
        n_cutoffs,
        figsize=(5 * n_cutoffs, 5 * n_algorithms),
        sharey="row",
        squeeze=False,
    )

    # Color palette for different MIN_PARTICLES_PER_CELL values
    colors = sns.color_palette("husl", len(min_particles_values))
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

    for row_idx, algorithm in enumerate(algorithms):
        for col_idx, cutoff in enumerate(cutoffs):
            ax = axes[row_idx][col_idx]
            cutoff_results = results[algorithm][cutoff]

            for idx, min_particles in enumerate(min_particles_values):
                n_atoms_list = []
                means = []
                stds = []

                for n_atoms, atom_results in cutoff_results.items():
                    if min_particles in atom_results:
                        mean, std = atom_results[min_particles]
                        if not np.isnan(mean):
                            n_atoms_list.append(n_atoms)
                            means.append(mean)
                            stds.append(std)

                if n_atoms_list:
                    marker = markers[idx % len(markers)]
                    ax.errorbar(
                        n_atoms_list,
                        means,
                        yerr=stds,
                        label=f"{min_particles}",
                        color=colors[idx],
                        marker=marker,
                        markersize=6,
                        capsize=3,
                        linewidth=1.5,
                        alpha=0.8,
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")

            # Column titles on the top row
            if row_idx == 0:
                ax.set_title(f"Cutoff = {cutoff} Å", fontsize=12)

            # X-axis labels on the bottom row
            if row_idx == n_algorithms - 1:
                ax.set_xlabel("Number of atoms", fontsize=11)

            # Y-axis labels and algorithm annotation on the left column
            if col_idx == 0:
                ax.set_ylabel(f"{ALGORITHM_LABELS[algorithm]}\nTime (ms)", fontsize=11)

            # Legend on the right column
            if col_idx == n_cutoffs - 1:
                ax.legend(title="MIN_PARTICLES\nPER_CELL", fontsize=8, title_fontsize=9)

    fig.suptitle(
        "Scaling with VESIN_CUDA_MIN_PARTICLES_PER_CELL\n(by algorithm)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_speedup(
    results,
    min_particles_values,
    baseline=32,
    output_file="min_particles_per_cell_speedup.png",
):
    sns.set_theme(style="whitegrid")

    algorithms = list(results.keys())
    cutoffs = list(results[algorithms[0]].keys())
    n_algorithms = len(algorithms)
    n_cutoffs = len(cutoffs)

    fig, axes = plt.subplots(
        n_algorithms,
        n_cutoffs,
        figsize=(5 * n_cutoffs, 5 * n_algorithms),
        sharey="row",
        squeeze=False,
    )

    colors = sns.color_palette("husl", len(min_particles_values))
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

    for row_idx, algorithm in enumerate(algorithms):
        for col_idx, cutoff in enumerate(cutoffs):
            ax = axes[row_idx][col_idx]
            cutoff_results = results[algorithm][cutoff]

            # Get baseline times
            baseline_times = {}
            for n_atoms, atom_results in cutoff_results.items():
                if baseline in atom_results:
                    mean, _ = atom_results[baseline]
                    if not np.isnan(mean):
                        baseline_times[n_atoms] = mean

            for idx, min_particles in enumerate(min_particles_values):
                n_atoms_list = []
                speedups = []

                for n_atoms, atom_results in cutoff_results.items():
                    if min_particles in atom_results and n_atoms in baseline_times:
                        mean, _ = atom_results[min_particles]
                        if not np.isnan(mean):
                            n_atoms_list.append(n_atoms)
                            speedups.append(baseline_times[n_atoms] / mean)

                if n_atoms_list:
                    marker = markers[idx % len(markers)]
                    ax.plot(
                        n_atoms_list,
                        speedups,
                        label=f"{min_particles}",
                        color=colors[idx],
                        marker=marker,
                        markersize=6,
                        linewidth=1.5,
                        alpha=0.8,
                    )

            ax.axhline(
                y=1.0,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"baseline ({baseline})",
            )
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3, which="both")

            # Column titles on the top row
            if row_idx == 0:
                ax.set_title(f"Cutoff = {cutoff} Å", fontsize=12)

            # X-axis labels on the bottom row
            if row_idx == n_algorithms - 1:
                ax.set_xlabel("Number of atoms", fontsize=11)

            # Y-axis labels on the left column
            if col_idx == 0:
                ax.set_ylabel(
                    f"{ALGORITHM_LABELS[algorithm]}\nSpeedup vs baseline ({baseline})",
                    fontsize=11,
                )

            # Legend on the right column
            if col_idx == n_cutoffs - 1:
                ax.legend(title="MIN_PARTICLES\nPER_CELL", fontsize=8, title_fontsize=9)

    fig.suptitle(
        "Speedup with different VESIN_CUDA_MIN_PARTICLES_PER_CELL\n(by algorithm)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    n_atoms_list = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    cutoffs = [3.0, 6.0, 12.0]
    min_particles_values = [4, 8, 16, 32, 64, 128, 256, 512]
    density = 0.1

    print("Benchmark: VESIN_CUDA_MIN_PARTICLES_PER_CELL")
    print("=" * 60)
    print(f"Atom counts: {n_atoms_list}")
    print(f"Cutoffs: {cutoffs}")
    print(f"Algorithms: {[ALGORITHM_LABELS[a] for a in ALGORITHMS]}")
    print(f"MIN_PARTICLES_PER_CELL values: {min_particles_values}")
    print(f"Density: {density}")
    print(f"Warmup: 10 iterations")
    print(f"Measurement: 25 iterations")

    results = run_benchmarks(
        n_atoms_list,
        cutoffs,
        min_particles_values,
        algorithms=ALGORITHMS,
        density=density,
        n_warmup=10,
        n_runs=25,
    )

    print_summary(results, min_particles_values)

    # Generate plots
    plot_results(results, min_particles_values)
    plot_speedup(results, min_particles_values, baseline=32)
