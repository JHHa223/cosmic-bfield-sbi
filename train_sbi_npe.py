#!/usr/bin/env python3
"""
Train an amortized Neural Posterior Estimator (NPE) for B(t) curves.

Expected dataset files
----------------------
B_curves.npy      shape (N, 512)
parameters.npy    shape (N, 4)
aux.npy           shape (N, 7)
tau_grid.npy      shape (512,)
metadata.json

SBI coordinates
---------------
theta = [
    log10(|Delta_0|),
    log10(|deltaDelta/Delta_0|),
    log10(Gamma_d/Gamma_c^(inst)),
    log10(R_coll),
]

where R_coll = nu_ii / nu_scatt,0 = aux[:, 0].

Observation
-----------
x(t) = log10(B(t)/B0) / x_scale

The script makes a fixed 90k/10k development/test split by default.
The held-out test indices are saved and are never passed to sbi training.
"""

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform


PARAMETER_NAMES = [
    "log10_abs_Delta0",
    "log10_abs_deltaDelta_over_Delta0",
    "log10_Gamma_d_over_Gamma_c_inst",
    "log10_R_coll",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CausalCNN + NSF Neural Posterior Estimator."
    )

    p.add_argument(
        "--data-dir",
        type=str,
        default="bfield_dataset_100k",
        help="Directory containing B_curves.npy, parameters.npy, aux.npy, tau_grid.npy.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="sbi_npe_run",
        help="Output directory for model, split indices, config, and training history.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="Number of completely held-out curves.",
    )

    # Observation preprocessing
    p.add_argument(
        "--x-scale",
        type=float,
        default=10.5,
        help="Scale applied after log10(B/B0). For this dataset max log10(B/B0) is 10.5.",
    )

    # Causal CNN
    p.add_argument("--cnn-layers", type=int, default=4)
    p.add_argument("--pool-kernel", type=int, default=4)
    p.add_argument("--embedding-dim", type=int, default=64)

    # NSF
    p.add_argument("--hidden-features", type=int, default=128)
    p.add_argument("--num-transforms", type=int, default=8)
    p.add_argument("--num-bins", type=int, default=10)

    # Training
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--stop-after-epochs", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--clip-max-norm", type=float, default=5.0)

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="PyTorch DataLoader workers used by sbi.",
    )
    p.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Regenerate the dev/test split even if split_indices.npz exists.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms. Can be slower.",
    )

    return p.parse_args()


def choose_device(device_arg):
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def set_seeds(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_dataset(data_dir):
    data_dir = Path(data_dir)

    b_path = data_dir / "B_curves.npy"
    p_path = data_dir / "parameters.npy"
    a_path = data_dir / "aux.npy"
    t_path = data_dir / "tau_grid.npy"
    m_path = data_dir / "metadata.json"

    for path in [b_path, p_path, a_path, t_path]:
        require_file(path)

    B = np.load(b_path, mmap_mode="r")
    P = np.load(p_path, mmap_mode="r")
    A = np.load(a_path, mmap_mode="r")
    tau = np.load(t_path)

    metadata = None
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if B.ndim != 2:
        raise ValueError(f"B_curves.npy must be 2D, got {B.shape}")

    n_samples, n_time = B.shape

    if P.shape != (n_samples, 4):
        raise ValueError(f"Unexpected parameters.npy shape: {P.shape}")

    if A.shape[0] != n_samples or A.shape[1] < 1:
        raise ValueError(f"Unexpected aux.npy shape: {A.shape}")

    if tau.shape != (n_time,):
        raise ValueError(
            f"tau_grid.npy has shape {tau.shape}, but B has {n_time} time points."
        )

    return B, P, A, tau, metadata


def make_or_load_split(n_samples, test_size, seed, out_dir, overwrite=False):
    split_path = Path(out_dir) / "split_indices.npz"

    if split_path.exists() and not overwrite:
        split = np.load(split_path)
        dev_idx = split["dev_idx"]
        test_idx = split["test_idx"]

        if len(dev_idx) + len(test_idx) != n_samples:
            raise ValueError(
                "Existing split_indices.npz does not match the current dataset size."
            )

        print(f"Reusing existing split: {split_path}")
        return dev_idx, test_idx

    if test_size <= 0 or test_size >= n_samples:
        raise ValueError("test_size must satisfy 0 < test_size < n_samples")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    test_idx = perm[:test_size].astype(np.int64)
    dev_idx = perm[test_size:].astype(np.int64)

    np.savez(
        split_path,
        dev_idx=dev_idx,
        test_idx=test_idx,
        seed=np.int64(seed),
    )

    print(f"Saved new fixed split: {split_path}")

    return dev_idx, test_idx


def build_theta_log(P, A, indices):
    """
    Build independent inference coordinates in log10 space.

    P[:, 0] = |Delta_0|
    P[:, 1] = |deltaDelta/Delta_0|
    P[:, 2] = Gamma_d/Gamma_c^(inst)
    A[:, 0] = R_coll = nu_ii/nu_scatt,0
    """

    theta_phys = np.column_stack(
        [
            np.asarray(P[indices, 0], dtype=np.float32),
            np.asarray(P[indices, 1], dtype=np.float32),
            np.asarray(P[indices, 2], dtype=np.float32),
            np.asarray(A[indices, 0], dtype=np.float32),
        ]
    )

    if not np.all(np.isfinite(theta_phys)):
        raise ValueError("Non-finite values found in theta_phys.")

    if np.any(theta_phys <= 0.0):
        raise ValueError("All physical SBI coordinates must be strictly positive.")

    theta_log = np.log10(theta_phys).astype(np.float32)

    return theta_log


def build_x(B, indices, x_scale):
    """
    Copy selected B curves into RAM, then transform in place:
        x = log10(B/B0) / x_scale
    """
    if x_scale <= 0.0:
        raise ValueError("x_scale must be positive.")

    # Fancy indexing a memmap returns an in-memory array.
    x = np.array(B[indices], dtype=np.float32, copy=True)

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values found in selected B curves.")

    if np.any(x <= 0.0):
        raise ValueError("B/B0 must be strictly positive before log10.")

    np.log10(x, out=x)
    x /= np.float32(x_scale)

    return x


def validate_prior_support(theta_log, low, high, tolerance=2e-5):
    low_np = np.asarray(low, dtype=np.float32)
    high_np = np.asarray(high, dtype=np.float32)

    if np.any(theta_log < (low_np - tolerance)):
        mins = theta_log.min(axis=0)
        raise ValueError(
            f"Training theta falls below prior. theta minima={mins}, prior low={low_np}"
        )

    if np.any(theta_log > (high_np + tolerance)):
        maxs = theta_log.max(axis=0)
        raise ValueError(
            f"Training theta exceeds prior. theta maxima={maxs}, prior high={high_np}"
        )


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def save_training_summary(inference, out_dir):
    out_dir = Path(out_dir)

    summary = inference.summary

    with open(out_dir / "training_summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    # Save a simple loss plot when these standard fields exist.
    try:
        train_loss = np.asarray(summary["training_loss"], dtype=float)
        val_loss = np.asarray(summary["validation_loss"], dtype=float)

        plt.figure(figsize=(7, 5))
        plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Training")
        plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Warning: could not make loss curve: {exc}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args.seed, deterministic=args.deterministic)

    device = choose_device(args.device)

    print("=" * 72)
    print("B-field SBI training: CausalCNN + NSF + NPE")
    print("=" * 72)
    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")

    try:
        import sbi
        print(f"sbi          : {sbi.__version__}")
    except Exception:
        pass

    print(f"Device       : {device}")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        print(f"GPU          : {torch.cuda.get_device_name(torch.device(device))}")

    print()

    B, P, A, tau, dataset_metadata = load_dataset(args.data_dir)

    n_samples, n_time = B.shape

    print(f"B shape      : {B.shape}")
    print(f"P shape      : {P.shape}")
    print(f"A shape      : {A.shape}")
    print(f"tau shape    : {tau.shape}")

    dev_idx, test_idx = make_or_load_split(
        n_samples=n_samples,
        test_size=args.test_size,
        seed=args.seed,
        out_dir=out_dir,
        overwrite=args.overwrite_split,
    )

    print(f"Development  : {len(dev_idx)}")
    print(f"Held-out test: {len(test_idx)}")
    print()

    # --------------------------------------------------------
    # Preprocess development set only.
    # Held-out test data are not passed to NPE.
    # --------------------------------------------------------

    print("Preparing development theta...")
    theta_dev_np = build_theta_log(P, A, dev_idx)

    print("Preparing development B(t)...")
    x_dev_np = build_x(B, dev_idx, args.x_scale)

    print(f"theta_dev    : {theta_dev_np.shape}, {theta_dev_np.dtype}")
    print(f"x_dev        : {x_dev_np.shape}, {x_dev_np.dtype}")

    # Physics/sampling prior in log10 coordinates.
    prior_low = np.array(
        [
            -4.0,                 # log10 |Delta_0|
            -6.0,                 # log10 perturb_ratio
            -4.0,                 # log10 Gamma_d/Gamma_c
            -2.0,                 # log10 R_coll
        ],
        dtype=np.float32,
    )

    prior_high = np.array(
        [
            -1.0,
            -1.0,
            2.0,
            np.log10(0.9),
        ],
        dtype=np.float32,
    )

    validate_prior_support(theta_dev_np, prior_low, prior_high)

    # CPU resident simulation pairs. sbi moves batches to GPU as needed.
    theta_dev = torch.from_numpy(theta_dev_np)
    x_dev = torch.from_numpy(x_dev_np)

    # Prior must live on the inference/training device.
    prior = BoxUniform(
        low=torch.tensor(prior_low, dtype=torch.float32, device=device),
        high=torch.tensor(prior_high, dtype=torch.float32, device=device),
    )

    # --------------------------------------------------------
    # Time-series embedding
    # --------------------------------------------------------

    embedding_net = embedding_nets.CausalCNNEmbedding(
        input_shape=(n_time,),
        num_conv_layers=args.cnn_layers,
        pool_kernel_size=args.pool_kernel,
        output_dim=args.embedding_dim,
    )

    # --------------------------------------------------------
    # Conditional density estimator: Neural Spline Flow
    # --------------------------------------------------------

    density_estimator_builder = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        num_bins=args.num_bins,
    )

    inference = NPE(
        prior=prior,
        density_estimator=density_estimator_builder,
        device=device,
    )

    dataloader_kwargs = {
        "num_workers": args.dataloader_workers,
        "pin_memory": bool(device.startswith("cuda")),
    }

    if args.dataloader_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training configuration")
    print("----------------------")
    print(f"CNN layers           : {args.cnn_layers}")
    print(f"pool kernel          : {args.pool_kernel}")
    print(f"embedding dim        : {args.embedding_dim}")
    print(f"flow                 : NSF")
    print(f"hidden features      : {args.hidden_features}")
    print(f"flow transforms      : {args.num_transforms}")
    print(f"spline bins          : {args.num_bins}")
    print(f"batch size           : {args.batch_size}")
    print(f"learning rate        : {args.learning_rate}")
    print(f"validation fraction  : {args.validation_fraction}")
    print(f"early-stop patience  : {args.stop_after_epochs}")
    print(f"max epochs           : {args.max_epochs}")
    print()

    t0 = time.time()

    inference = inference.append_simulations(
        theta_dev,
        x_dev,
        data_device="cpu",
    )

    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_epochs,
        clip_max_norm=args.clip_max_norm,
        show_train_summary=True,
        dataloader_kwargs=dataloader_kwargs,
    )

    elapsed = time.time() - t0

    posterior = inference.build_posterior(
        density_estimator=density_estimator,
        sample_with="direct",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Saving outputs...")

    # Official sbi objects are picklable. Keep both for later evaluation.
    with open(out_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    with open(out_dir / "inference.pkl", "wb") as f:
        pickle.dump(inference, f)

    # Also save a CPU state_dict as an additional portable checkpoint.
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in density_estimator.state_dict().items()
    }
    torch.save(cpu_state_dict, out_dir / "density_estimator_state_dict.pt")

    np.save(out_dir / "tau_grid.npy", tau.astype(np.float32))

    save_training_summary(inference, out_dir)

    run_config = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "seed": args.seed,
        "device": device,
        "n_samples_total": int(n_samples),
        "n_development": int(len(dev_idx)),
        "n_test": int(len(test_idx)),
        "n_time": int(n_time),
        "theta_parameter_names": PARAMETER_NAMES,
        "prior_low_log10": prior_low.tolist(),
        "prior_high_log10": prior_high.tolist(),
        "x_transform": "log10(B/B0) / x_scale",
        "x_scale": float(args.x_scale),
        "cnn_layers": args.cnn_layers,
        "pool_kernel": args.pool_kernel,
        "embedding_dim": args.embedding_dim,
        "density_estimator": "nsf",
        "hidden_features": args.hidden_features,
        "num_transforms": args.num_transforms,
        "num_bins": args.num_bins,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "stop_after_epochs": args.stop_after_epochs,
        "max_epochs": args.max_epochs,
        "clip_max_norm": args.clip_max_norm,
        "dataloader_workers": args.dataloader_workers,
        "elapsed_seconds": elapsed,
        "dataset_metadata": dataset_metadata,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }

    try:
        import sbi
        run_config["sbi_version"] = sbi.__version__
    except Exception:
        pass

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print()
    print("=" * 72)
    print("Training finished")
    print("=" * 72)
    print(f"Elapsed              : {elapsed / 60.0:.2f} min")
    print(f"Posterior             : {out_dir / 'posterior.pkl'}")
    print(f"Inference object      : {out_dir / 'inference.pkl'}")
    print(f"State dict            : {out_dir / 'density_estimator_state_dict.pt'}")
    print(f"Fixed split           : {out_dir / 'split_indices.npz'}")
    print(f"Run configuration     : {out_dir / 'run_config.json'}")
    print(f"Training summary      : {out_dir / 'training_summary.json'}")
    print(f"Loss curve            : {out_dir / 'loss_curve.png'}")
    print()

    # Deliberately do not evaluate on test data here.
    # The held-out set will be used by evaluate_sbi.py.


if __name__ == "__main__":
    main()
