#!/usr/bin/env python3
"""
Evaluate a trained amortized SBI posterior on a held-out B(t) curve.

Matched to train_sbi_npe.py and make_B_dataset_parallel.py.

Outputs for one held-out test case
----------------------------------
observation_curve.png
marginal_posteriors_log.png
corner_posterior_log.png
posterior_predictive.png
posterior_summary.json
posterior_samples_log.npy
posterior_samples_physical.npy
posterior_predictive_curves.npy
"""

import argparse
import json
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


LOG_LABELS = [
    r"$\log_{10}|\Delta_0|$",
    r"$\log_{10}|\delta\Delta/\Delta_0|$",
    r"$\log_{10}(\Gamma_d/\Gamma_c^{(\mathrm{inst})})$",
    r"$\log_{10}R_{\mathrm{coll}}$",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NPE on a fixed held-out B(t) curve."
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--test-position", type=int, default=0)
    p.add_argument("--n-posterior", type=int, default=20000)
    p.add_argument("--n-ppc", type=int, default=200)
    p.add_argument("--ppc-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    for name in ["B_curves.npy", "parameters.npy", "aux.npy", "tau_grid.npy"]:
        require_file(data_dir / name)

    B = np.load(data_dir / "B_curves.npy", mmap_mode="r")
    P = np.load(data_dir / "parameters.npy", mmap_mode="r")
    A = np.load(data_dir / "aux.npy", mmap_mode="r")
    tau = np.load(data_dir / "tau_grid.npy")

    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else None
    return B, P, A, tau, metadata


def build_true_theta(P, A, dataset_index):
    theta_phys4 = np.array(
        [
            P[dataset_index, 0],
            P[dataset_index, 1],
            P[dataset_index, 2],
            A[dataset_index, 0],
        ],
        dtype=np.float64,
    )
    if np.any(theta_phys4 <= 0):
        raise ValueError("True SBI coordinates must be strictly positive.")
    return np.log10(theta_phys4), theta_phys4


def preprocess_observation(B_curve, x_scale):
    x = np.array(B_curve, dtype=np.float32, copy=True)
    if np.any(x <= 0):
        raise ValueError("B/B0 must be positive.")
    np.log10(x, out=x)
    x /= np.float32(x_scale)
    return x


def posterior_samples_to_physical(samples_log, beta0):
    physical4 = 10.0 ** np.asarray(samples_log, dtype=np.float64)
    D = physical4[:, 0]
    Rcoll = physical4[:, 3]
    excess = np.maximum(D - 2.0 / beta0, 0.0)
    nu_ratio = Rcoll * excess**1.5
    return np.column_stack([physical4, nu_ratio])


def summarize_1d(samples, truth):
    q2p5, q16, q50, q84, q97p5 = np.percentile(
        samples, [2.5, 16.0, 50.0, 84.0, 97.5]
    )
    return {
        "truth": float(truth),
        "median": float(q50),
        "q16": float(q16),
        "q84": float(q84),
        "q2p5": float(q2p5),
        "q97p5": float(q97p5),
        "inside_68": bool(q16 <= truth <= q84),
        "inside_95": bool(q2p5 <= truth <= q97p5),
        "posterior_percentile_of_truth": float(np.mean(samples <= truth)),
    }


def make_observation_plot(tau, B_true, out_path):
    plt.figure(figsize=(7, 5))
    plt.loglog(tau, B_true, linewidth=2)
    plt.xlabel(r"$t\Omega_{i,0}$")
    plt.ylabel(r"$B/B_0$")
    plt.title("Held-out B(t) observation")
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_marginal_plot(samples_log, theta_true_log, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for j, ax in enumerate(axes):
        ax.hist(samples_log[:, j], bins=60, density=True, alpha=0.8)
        ax.axvline(theta_true_log[j], linestyle="--", linewidth=2, label="Truth")

        q16, q50, q84 = np.percentile(samples_log[:, j], [16.0, 50.0, 84.0])
        ax.axvline(q50, linewidth=1.5, label="Posterior median")
        ax.axvspan(q16, q84, alpha=0.15)

        ax.set_xlabel(LOG_LABELS[j])
        ax.set_ylabel("Posterior density")
        ax.tick_params(direction="in", top=True, right=True)
        if j == 0:
            ax.legend()

    fig.suptitle("Marginal posterior distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_corner_plot(samples_log, theta_true_log, out_path, max_points=8000):
    rng = np.random.default_rng(12345)
    if len(samples_log) > max_points:
        idx = rng.choice(len(samples_log), size=max_points, replace=False)
        S = samples_log[idx]
    else:
        S = samples_log

    ndim = S.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(11, 11))

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(S[:, col], bins=50, density=True, alpha=0.8)
                ax.axvline(theta_true_log[col], linestyle="--", linewidth=1.5)
            else:
                ax.hist2d(S[:, col], S[:, row], bins=55)
                ax.plot(
                    theta_true_log[col],
                    theta_true_log[row],
                    marker="x",
                    markersize=8,
                    color="red",
                    linestyle="none",
                )

            if row == ndim - 1:
                ax.set_xlabel(LOG_LABELS[col],fontsize=18)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(LOG_LABELS[row],fontsize=18)
            elif col != 0:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=18,direction="in", top=True, right=True)

    #fig.suptitle("Posterior parameter degeneracies")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Forward model reproduced from make_B_dataset_parallel.py for PPC.
# -----------------------------------------------------------------------------

def generate_B_curve_for_ppc(task):
    theta_phys, cfg = task

    D = abs(float(theta_phys[0]))
    eps = abs(float(theta_phys[1]))
    r_gamma = abs(float(theta_phys[2]))
    r_coll = abs(float(theta_phys[3]))

    beta0 = float(cfg["beta0"])
    tau_output_min = float(cfg["tau_output_min"])
    tau_output_max = float(cfg["tau_output_max"])
    n_time = int(cfg["n_time"])
    rtol = float(cfg.get("rtol", 1e-7))
    atol = float(cfg.get("atol", 1e-9))

    dD = eps * D
    excess0 = max(D - 2.0 / beta0, 0.0)
    if excess0 <= 0:
        return None

    nu_scatt0_hat = excess0**1.5
    nu_hat = r_coll * nu_scatt0_hat

    gamma_c_hat = D**1.5 + 2.0 * np.exp(-0.5 * r_gamma) * dD**1.5
    gamma_d_hat = r_gamma * gamma_c_hat

    b_sat = np.sqrt(beta0 * D / 2.0)
    y_sat = np.log(b_sat)

    perturb_drive0 = 2.0 * np.exp(-0.5 * r_gamma) * dD**1.5
    initial_growth = (
        -nu_hat + excess0**1.5 + perturb_drive0
    ) / (1.0 + D)

    if initial_growth <= 0:
        return None

    def rhs(tau, y):
        b = np.exp(y[0])
        beta = beta0 / (b * b)
        instability_excess = max(D - 2.0 / beta, 0.0)
        main_drive = instability_excess**1.5
        perturb_drive = (
            2.0
            * np.exp(-gamma_d_hat * tau - 0.5 * r_gamma)
            * dD**1.5
        )
        db_dtau = (
            -nu_hat + (main_drive + perturb_drive) * b * b
        ) / (1.0 + D)
        return [db_dtau / b]

    def saturation_event(tau, y):
        return y[0] - y_sat

    saturation_event.terminal = True
    saturation_event.direction = 1

    sol = solve_ivp(
        rhs,
        (0.0, tau_output_max),
        y0=[0.0],
        method="Radau",
        rtol=rtol,
        atol=atol,
        max_step=max(1.0, tau_output_max / 1000.0),
        events=saturation_event,
        dense_output=True,
    )

    if not sol.success:
        return None

    tau = np.logspace(
        np.log10(tau_output_min), np.log10(tau_output_max), n_time
    )

    if len(sol.t_events[0]) > 0:
        tau_sat = float(sol.t_events[0][0])
        b = np.empty_like(tau)
        before = tau <= tau_sat
        if np.any(before):
            b[before] = np.exp(sol.sol(tau[before])[0])
        b[~before] = b_sat
    else:
        b = np.exp(sol.sol(tau)[0])

    b = np.minimum(b, b_sat)
    if not np.all(np.isfinite(b)):
        return None

    return b.astype(np.float32)


def run_ppc(samples_log, model_config, n_ppc, workers, seed):
    rng = np.random.default_rng(seed)
    n_ppc = min(n_ppc, len(samples_log))
    chosen = rng.choice(len(samples_log), size=n_ppc, replace=False)
    theta_ppc = 10.0 ** samples_log[chosen]

    tasks = [(theta_ppc[i], model_config) for i in range(n_ppc)]

    if workers <= 1:
        curves = [generate_B_curve_for_ppc(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            curves = list(
                pool.imap(
                    generate_B_curve_for_ppc,
                    tasks,
                    chunksize=max(1, n_ppc // (workers * 4)),
                )
            )

    valid = [curve for curve in curves if curve is not None]
    if not valid:
        raise RuntimeError("No valid posterior-predictive curves were generated.")

    return np.stack(valid, axis=0)

def make_ppc_plot(tau, B_true, B_ppc, out_path, show_legend=False):
    q2p5 = np.percentile(B_ppc, 2.5, axis=0)
    q16 = np.percentile(B_ppc, 16.0, axis=0)
    q50 = np.percentile(B_ppc, 50.0, axis=0)
    q84 = np.percentile(B_ppc, 84.0, axis=0)
    q97p5 = np.percentile(B_ppc, 97.5, axis=0)

    plt.figure(figsize=(8, 6))

    plt.fill_between(
        tau, q2p5, q97p5,
        alpha=0.15,
        label="95% posterior predictive"
    )

    plt.fill_between(
        tau, q16, q84,
        alpha=0.30,
        label="68% posterior predictive"
    )

    plt.loglog(
        tau, q50,
        linewidth=2,
        label="Posterior predictive median"
    )

    plt.loglog(
        tau, B_true,
        linestyle="--",
        linewidth=2,
        label="True B(t)"
    )

    plt.xlabel(r"$t\Omega_{i,0}$", fontsize=18)
    plt.ylabel(r"$B/B_0$", fontsize=18)
    plt.xlim(10, 1e5)

    if show_legend:
        plt.legend(fontsize=18, loc="upper left")

    plt.tick_params(
        labelsize=18,
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "median_log10_rmse_dex": float(
            np.sqrt(np.mean((np.log10(q50) - np.log10(B_true)) ** 2))
        ),
        "fraction_timepoints_true_inside_68_band": float(
            np.mean((B_true >= q16) & (B_true <= q84))
        ),
        "fraction_timepoints_true_inside_95_band": float(
            np.mean((B_true >= q2p5) & (B_true <= q97p5))
        ),
    }

def main():
    args = parse_args()
    set_seeds(args.seed)

    run_dir = Path(args.run_dir)
    require_file(run_dir / "posterior.pkl")
    require_file(run_dir / "split_indices.npz")
    require_file(run_dir / "run_config.json")

    run_config = load_json(run_dir / "run_config.json")
    data_dir = Path(args.data_dir) if args.data_dir else Path(run_config["data_dir"])

    B, P, A, tau, dataset_metadata = load_dataset(data_dir)

    split = np.load(run_dir / "split_indices.npz")
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)

    if args.test_position < 0 or args.test_position >= len(test_idx):
        raise IndexError(
            f"test-position must be 0..{len(test_idx)-1}, got {args.test_position}"
        )

    dataset_index = int(test_idx[args.test_position])

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = run_dir / "evaluation" / f"test_{args.test_position:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_scale = float(run_config["x_scale"])

    if dataset_metadata is not None:
        model_config = dataset_metadata["model_config"]
    else:
        model_config = run_config["dataset_metadata"]["model_config"]

    beta0 = float(model_config["beta0"])

    B_true = np.array(B[dataset_index], dtype=np.float32, copy=True)
    theta_true_log, theta_true_phys4 = build_true_theta(P, A, dataset_index)
    true_nu_ratio = float(P[dataset_index, 3])
    theta_true_phys5 = np.concatenate([theta_true_phys4, [true_nu_ratio]])

    x_o_np = preprocess_observation(B_true, x_scale)

    print("=" * 72)
    print("SBI held-out evaluation")
    print("=" * 72)
    print(f"run directory      : {run_dir}")
    print(f"data directory     : {data_dir}")
    print(f"test position      : {args.test_position}")
    print(f"raw dataset index  : {dataset_index}")
    print(f"posterior samples  : {args.n_posterior}")
    print(f"PPC curves         : {args.n_ppc}")
    print()

    with open(run_dir / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)

    # Keep the conditioning observation on the same device
    # as the trained posterior estimator.
    try:
        posterior_device = next(
            posterior.posterior_estimator.parameters()
        ).device
    except (AttributeError, StopIteration):
        posterior_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if hasattr(posterior, "to"):
        posterior.to(posterior_device)

    x_o = (
        torch.from_numpy(x_o_np)
        .float()
        .to(posterior_device)
    )

    print(f"posterior device    : {posterior_device}")
    print(f"observation device  : {x_o.device}")

    print("Drawing posterior samples...")
    t0 = time.time()
    with torch.no_grad():
        samples_t = posterior.sample(
            (args.n_posterior,),
            x=x_o,
            show_progress_bars=True,
        )
    posterior_seconds = time.time() - t0

    samples_log = samples_t.detach().cpu().numpy().astype(np.float64)
    if samples_log.ndim != 2 or samples_log.shape[1] != 4:
        raise ValueError(f"Unexpected posterior sample shape: {samples_log.shape}")

    samples_phys5 = posterior_samples_to_physical(samples_log, beta0)

    np.save(out_dir / "posterior_samples_log.npy", samples_log.astype(np.float32))
    np.save(out_dir / "posterior_samples_physical.npy", samples_phys5.astype(np.float32))
    np.save(out_dir / "true_theta_log.npy", theta_true_log.astype(np.float32))
    np.save(out_dir / "true_theta_physical.npy", theta_true_phys5.astype(np.float32))

    make_observation_plot(tau, B_true, out_dir / "observation_curve.png")
    make_marginal_plot(samples_log, theta_true_log, out_dir / "marginal_posteriors_log.png")
    make_corner_plot(samples_log, theta_true_log, out_dir / "corner_posterior_log.png")

    print("Running posterior predictive simulations...")
    t1 = time.time()
    B_ppc = run_ppc(
        samples_log,
        model_config=model_config,
        n_ppc=args.n_ppc,
        workers=args.ppc_workers,
        seed=args.seed + 1,
    )
    ppc_seconds = time.time() - t1

    np.save(out_dir / "posterior_predictive_curves.npy", B_ppc.astype(np.float32))

    # --------------------------------------------------------
    # Show the PPC legend only for the low-Delta0 / Regime-A case.
    #
    # The grid-selection script writes evaluation directories as
    #     low_A, low_B, ..., high_C
    # so we can infer the panel identity directly from out_dir.name.
    # For ordinary single-case evaluations whose directory name does
    # not follow this convention, the legend is simply omitted.
    # --------------------------------------------------------
    case_parts = out_dir.name.split("_")

    if len(case_parts) >= 2:
        delta_level = case_parts[0].lower()
        regime = case_parts[1].upper()
    else:
        delta_level = None
        regime = None

    show_legend = (
        delta_level == "low"
        and regime == "A"
    )

    print(
        f"PPC panel identity   : delta_level={delta_level}, "
        f"regime={regime}, show_legend={show_legend}"
    )

    ppc_metrics = make_ppc_plot(
        tau,
        B_true,
        B_ppc,
        out_dir / "posterior_predictive.png",
        show_legend=show_legend,
    )

    log_names = [
        "log10_abs_Delta0",
        "log10_abs_deltaDelta_over_Delta0",
        "log10_Gamma_d_over_Gamma_c_inst",
        "log10_R_coll",
    ]
    phys_names = [
        "abs_Delta0",
        "abs_deltaDelta_over_Delta0",
        "Gamma_d_over_Gamma_c_inst",
        "R_coll",
        "nu_ii_over_Omega_i0",
    ]

    summary = {
        "test_position": int(args.test_position),
        "dataset_index": dataset_index,
        "n_posterior_samples": int(args.n_posterior),
        "posterior_sampling_seconds": float(posterior_seconds),
        "n_requested_ppc": int(args.n_ppc),
        "n_valid_ppc": int(B_ppc.shape[0]),
        "ppc_seconds": float(ppc_seconds),
        "x_scale": x_scale,
        "log_parameter_summary": {},
        "physical_parameter_summary": {},
        "posterior_predictive_metrics": ppc_metrics,
    }

    for j, name in enumerate(log_names):
        summary["log_parameter_summary"][name] = summarize_1d(
            samples_log[:, j], theta_true_log[j]
        )

    for j, name in enumerate(phys_names):
        summary["physical_parameter_summary"][name] = summarize_1d(
            samples_phys5[:, j], theta_true_phys5[j]
        )

    with open(out_dir / "posterior_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Posterior medians and 68% intervals")
    print("-----------------------------------")
    for j, name in enumerate(phys_names):
        q16, q50, q84 = np.percentile(samples_phys5[:, j], [16, 50, 84])
        truth = theta_true_phys5[j]
        print(
            f"{name:36s} truth={truth:.6e}  "
            f"median={q50:.6e}  68%=[{q16:.6e}, {q84:.6e}]"
        )

    print()
    print("Posterior predictive")
    print("--------------------")
    for key, value in ppc_metrics.items():
        print(f"{key:48s}: {value:.6e}")

    print()
    print(f"Saved evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
