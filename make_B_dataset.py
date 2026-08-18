#!/usr/bin/env python3
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.lib.format import open_memmap


# ============================================================
# Magnetic-field evolution model
#
# Normalized variables:
#   b   = B / B0
#   tau = Omega_i,0 * t
#
# Physics-informed collision parameterization:
#   R_coll = nu_ii / nu_scatt,0
#
# with
#   nu_scatt,0 / Omega_i,0
#       = (|Delta_0| - 2/beta0)^(3/2)
#
# following the scattering-rate scaling used in Eq. (43).
# ============================================================

def log_uniform(rng, low, high):
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high))


def generate_B_curve(
    delta0,
    perturb_ratio,
    gamma_ratio,
    r_coll,
    beta0=2.0e22,
    tau_output_min=1.0,
    tau_output_max=1.0e6,
    n_time=512,
    rtol=1.0e-7,
    atol=1.0e-9,
):
    """
    Generate one physically admissible, non-decreasing B(t) curve.

    Parameters
    ----------
    delta0 : float
        |Delta_0|
    perturb_ratio : float
        |deltaDelta / Delta_0|
    gamma_ratio : float
        Gamma_d / Gamma_c^(inst)
    r_coll : float
        nu_ii / nu_scatt,0

    beta0 : float
        Fixed initial plasma beta.
    tau_output_min, tau_output_max : float
        Returned time range in tau = Omega_i,0 t.
        IMPORTANT: integration itself starts at tau=0.
    n_time : int
        Number of logarithmically spaced output points.

    Returns
    -------
    tau : ndarray
    b : ndarray
        B/B0
    info : dict
        Derived quantities.
    """

    D = abs(float(delta0))
    eps = abs(float(perturb_ratio))
    r_gamma = abs(float(gamma_ratio))
    r_coll = abs(float(r_coll))

    dD = eps * D

    # --------------------------------------------------------
    # Initial scattering rate from Eq. (43):
    #
    # nu_scatt / Omega_i ~ (|Delta| - 2 beta^{-1})^(3/2)
    #
    # Evaluate at B=B0, beta=beta0.
    # --------------------------------------------------------
    excess0 = max(D - 2.0 / beta0, 0.0)

    if excess0 <= 0.0:
        raise ValueError(
            "Initial state is not in the unstable domain: "
            "|Delta_0| <= 2/beta0."
        )

    nu_scatt0_hat = excess0 ** 1.5

    # Instead of independently drawing nu_ii/Omega_i,0 over a huge box,
    # derive it from the physically scaled collision strength.
    nu_hat = r_coll * nu_scatt0_hat

    # --------------------------------------------------------
    # Characteristic instability rate from the Eq. (49) scaling.
    # gamma_c_hat = Gamma_c^(inst) / Omega_i,0
    # --------------------------------------------------------
    gamma_c_hat = (
        D ** 1.5
        + 2.0 * np.exp(-0.5 * r_gamma) * dD ** 1.5
    )

    gamma_d_hat = r_gamma * gamma_c_hat

    # --------------------------------------------------------
    # Quasi-stability saturation:
    # beta_sat ~ 2 / |Delta_0|
    #
    # beta ~ beta0 / b^2  ->  b_sat = sqrt(beta0*D/2)
    # --------------------------------------------------------
    b_sat = np.sqrt(beta0 * D / 2.0)
    y_sat = np.log(b_sat)

    # Initial dimensionless growth rate db/dtau at b=1.
    perturb_drive0 = (
        2.0
        * np.exp(-0.5 * r_gamma)
        * dD ** 1.5
    )

    initial_growth = (
        -nu_hat
        + excess0 ** 1.5
        + perturb_drive0
    ) / (1.0 + D)

    if initial_growth <= 0.0:
        raise ValueError(
            "Non-amplifying initial condition: db/dtau <= 0. "
            "Decrease r_coll or change the sampled parameters."
        )

    # --------------------------------------------------------
    # Integrate y = ln(b), not b, for numerical stability.
    #
    # IMPORTANT:
    # integration starts at tau=0 with B/B0=1.
    # The logarithmic output grid begins later at tau_output_min.
    # --------------------------------------------------------
    def rhs(tau, y):
        b = np.exp(y[0])

        # Approximate beta evolution for fixed thermal pressure.
        beta = beta0 / (b * b)

        instability_excess = max(D - 2.0 / beta, 0.0)
        main_drive = instability_excess ** 1.5

        perturb_drive = (
            2.0
            * np.exp(
                -gamma_d_hat * tau
                - 0.5 * r_gamma
            )
            * dD ** 1.5
        )

        # Normalized form of Eq. (45).
        db_dtau = (
            -nu_hat
            + (main_drive + perturb_drive) * b * b
        ) / (1.0 + D)

        return [db_dtau / b]

    def saturation_event(tau, y):
        return y[0] - y_sat

    saturation_event.terminal = True
    saturation_event.direction = 1

    sol = solve_ivp(
        rhs,
        (0.0, tau_output_max),
        y0=[0.0],  # B(0)/B0 = 1
        method="Radau",
        rtol=rtol,
        atol=atol,
        max_step=max(1.0, tau_output_max / 1000.0),
        events=saturation_event,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    tau = np.logspace(
        np.log10(tau_output_min),
        np.log10(tau_output_max),
        n_time,
    )

    if len(sol.t_events[0]) > 0:
        tau_sat = float(sol.t_events[0][0])

        b = np.empty_like(tau)
        before = tau <= tau_sat

        if np.any(before):
            b[before] = np.exp(sol.sol(tau[before])[0])

        b[~before] = b_sat

    else:
        tau_sat = np.nan
        b = np.exp(sol.sol(tau)[0])

    # Numerical safety.
    b = np.minimum(b, b_sat)

    # Reject pathological curves.
    if not np.all(np.isfinite(b)):
        raise ValueError("Non-finite B(t) values.")

    if np.min(b) < 1.0 - 1.0e-6:
        raise ValueError("B(t) dropped below B0.")

    # Allow only tiny floating-point noise in the monotonicity check.
    db = np.diff(b)
    tolerance = 1.0e-7 * np.maximum(b[:-1], 1.0)

    if np.any(db < -tolerance):
        raise ValueError("Non-monotonic B(t) curve.")

    info = {
        "nu_ratio": nu_hat,
        "r_coll": r_coll,
        "nu_scatt0_hat": nu_scatt0_hat,
        "gamma_c_hat": gamma_c_hat,
        "gamma_d_hat": gamma_d_hat,
        "tau_sat": tau_sat,
        "b_sat": b_sat,
        "initial_growth": initial_growth,
    }

    return tau, b, info


# ============================================================
# Parameter sampling
# ============================================================

def sample_parameter_set(
    rng,
    delta0_range=(1.0e-4, 1.0e-1),
    perturb_ratio_range=(1.0e-6, 1.0e-1),
    gamma_ratio_range=(1.0e-4, 1.0e2),
    r_coll_range=(1.0e-2, 9.0e-1),
):
    """
    All four sampling coordinates are log-uniform.

    Note:
      nu_ii/Omega_i,0 is NOT independently sampled.
      It is derived from r_coll and nu_scatt,0.
    """

    delta0 = log_uniform(rng, *delta0_range)
    perturb_ratio = log_uniform(rng, *perturb_ratio_range)
    gamma_ratio = log_uniform(rng, *gamma_ratio_range)
    r_coll = log_uniform(rng, *r_coll_range)

    return delta0, perturb_ratio, gamma_ratio, r_coll


# ============================================================
# Dataset generation
# ============================================================

def build_dataset(
    n_samples=100,
    n_time=512,
    beta0=2.0e22,
    tau_output_min=1.0,
    tau_output_max=1.0e6,
    seed=42,
    out_dir="bfield_dataset",
    preview_n=20,
    progress_every=100,
):
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    B_path = os.path.join(out_dir, "B_curves.npy")
    P_path = os.path.join(out_dir, "parameters.npy")
    A_path = os.path.join(out_dir, "aux.npy")
    T_path = os.path.join(out_dir, "tau_grid.npy")

    # --------------------------------------------------------
    # parameters.npy columns:
    #   0: |Delta_0|
    #   1: |deltaDelta/Delta_0|
    #   2: Gamma_d/Gamma_c^(inst)
    #   3: nu_ii/Omega_i,0
    #
    # aux.npy columns:
    #   0: R_coll = nu_ii/nu_scatt,0
    #   1: nu_scatt,0/Omega_i,0
    #   2: Gamma_c^(inst)/Omega_i,0
    #   3: Gamma_d/Omega_i,0
    #   4: tau_sat
    #   5: B_sat/B0
    #   6: initial db/dtau
    # --------------------------------------------------------

    B_curves = open_memmap(
        B_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_time),
    )

    parameters = open_memmap(
        P_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, 4),
    )

    aux = open_memmap(
        A_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, 7),
    )

    tau_grid = None

    accepted = 0
    rejected = 0
    attempts = 0

    t0 = time.time()

    while accepted < n_samples:
        attempts += 1

        delta0, perturb_ratio, gamma_ratio, r_coll = sample_parameter_set(rng)

        try:
            tau, b, info = generate_B_curve(
                delta0=delta0,
                perturb_ratio=perturb_ratio,
                gamma_ratio=gamma_ratio,
                r_coll=r_coll,
                beta0=beta0,
                tau_output_min=tau_output_min,
                tau_output_max=tau_output_max,
                n_time=n_time,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            rejected += 1
            continue

        if tau_grid is None:
            tau_grid = tau.astype(np.float32)
            np.save(T_path, tau_grid)

        B_curves[accepted] = b.astype(np.float32)

        # Keep the original inference target nu_ii/Omega_i,0
        # in the fourth column.
        parameters[accepted] = np.array(
            [
                delta0,
                perturb_ratio,
                gamma_ratio,
                info["nu_ratio"],
            ],
            dtype=np.float32,
        )

        aux[accepted] = np.array(
            [
                info["r_coll"],
                info["nu_scatt0_hat"],
                info["gamma_c_hat"],
                info["gamma_d_hat"],
                info["tau_sat"],
                info["b_sat"],
                info["initial_growth"],
            ],
            dtype=np.float32,
        )

        accepted += 1

        if accepted % progress_every == 0 or accepted == n_samples:
            elapsed = time.time() - t0
            rate = accepted / elapsed

            if rate > 0:
                eta_hr = (n_samples - accepted) / rate / 3600.0
            else:
                eta_hr = np.nan

            print(
                f"[{accepted:>7d}/{n_samples}] "
                f"attempts={attempts}, "
                f"rejected={rejected}, "
                f"rate={rate:.2f} samples/s, "
                f"ETA={eta_hr:.2f} hr"
            )

            B_curves.flush()
            parameters.flush()
            aux.flush()

    B_curves.flush()
    parameters.flush()
    aux.flush()

    metadata = {
        "n_samples": int(n_samples),
        "n_time": int(n_time),
        "beta0": float(beta0),
        "tau_output_min": float(tau_output_min),
        "tau_output_max": float(tau_output_max),
        "seed": int(seed),
        "parameter_columns": [
            "|Delta_0|",
            "|deltaDelta/Delta_0|",
            "Gamma_d/Gamma_c^(inst)",
            "nu_ii/Omega_i,0",
        ],
        "aux_columns": [
            "R_coll=nu_ii/nu_scatt,0",
            "nu_scatt,0/Omega_i,0",
            "Gamma_c^(inst)/Omega_i,0",
            "Gamma_d/Omega_i,0",
            "tau_sat",
            "B_sat/B0",
            "initial_db_dtau",
        ],
        "sampling_ranges": {
            "|Delta_0|": [1.0e-4, 1.0e-1],
            "|deltaDelta/Delta_0|": [1.0e-6, 1.0e-1],
            "Gamma_d/Gamma_c^(inst)": [1.0e-4, 1.0e2],
            "R_coll": [1.0e-2, 9.0e-1],
        },
        "rejected_attempts": int(rejected),
        "total_attempts": int(attempts),
    }

    with open(
        os.path.join(out_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    make_preview(
        out_dir=out_dir,
        n_show=min(preview_n, n_samples),
    )

    elapsed = time.time() - t0

    print("\nFinished.")
    print(f"Accepted : {accepted}")
    print(f"Rejected : {rejected}")
    print(f"Elapsed  : {elapsed/60.0:.2f} min")
    print(f"Output   : {out_dir}")


# ============================================================
# Preview / diagnostics
# ============================================================

def make_preview(out_dir, n_show=20):
    tau = np.load(
        os.path.join(out_dir, "tau_grid.npy")
    )

    B_curves = np.load(
        os.path.join(out_dir, "B_curves.npy"),
        mmap_mode="r",
    )

    n_show = min(n_show, B_curves.shape[0])

    # Check all stored curves.
    min_B = np.min(B_curves)

    diff = np.diff(B_curves, axis=1)
    tol = 1.0e-6 * np.maximum(B_curves[:, :-1], 1.0)

    monotonic_fraction = np.mean(
        np.all(diff >= -tol, axis=1)
    )

    print("\nDataset diagnostics")
    print(f"minimum B/B0      = {min_B:.6e}")
    print(
        f"monotonic fraction = "
        f"{monotonic_fraction:.6f}"
    )

    plt.figure(figsize=(8, 6))

    for i in range(n_show):
        plt.loglog(
            tau,
            B_curves[i],
            linewidth=1.5,
            alpha=0.8,
        )

    plt.xlabel(
        r"$t\Omega_{i,0}$",
        fontsize=14,
    )

    plt.ylabel(
        r"$B/B_0$",
        fontsize=14,
    )

    plt.title(
        f"Preview of {n_show} physics-filtered B(t) curves"
    )

    plt.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    plt.tight_layout()

    fig_path = os.path.join(
        out_dir,
        "preview_curves.png",
    )

    plt.savefig(
        fig_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Preview figure: {fig_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help=(
            "Number of accepted curves. "
            "Use 100 for testing, then 100000 for the full dataset."
        ),
    )

    parser.add_argument(
        "--n-time",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="bfield_dataset",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--beta0",
        type=float,
        default=2.0e22,
    )

    parser.add_argument(
        "--tau-min",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tau-max",
        type=float,
        default=1.0e6,
    )

    args = parser.parse_args()

    build_dataset(
        n_samples=args.n_samples,
        n_time=args.n_time,
        beta0=args.beta0,
        tau_output_min=args.tau_min,
        tau_output_max=args.tau_max,
        seed=args.seed,
        out_dir=args.out_dir,
    )
