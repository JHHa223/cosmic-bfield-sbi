# Neural Simulation-Based Inference of Plasma Parameters from Cosmic Magnetic-Field Evolution

This repository contains the numerical forward model and simulation-based inference (SBI) pipeline used to infer plasma parameters from the time evolution of pressure-anisotropy-driven cosmic magnetic-field amplification.

## Workflow

The analysis consists of three main stages:

```text
Physical parameter sampling
        |
        v
Synthetic B-field histories
        |
        v
Neural posterior estimation
        |
        v
Posterior inference and posterior predictive checks
```

Run the complete workflow from the repository root with:

```bash
bash make_B_dataset.sh
bash train_sbi_npe.sh
bash evaluate_sbi.sh
```

## Repository structure

```text
.
├── README.md
├── make_B_dataset.py
├── make_B_dataset.sh
├── train_sbi_npe.py
├── train_sbi_npe.sh
├── evaluate_sbi.py
├── evaluate_sbi.sh
│
├── bfield_dataset_100k/      # generated synthetic data
│   ├── B_curves.npy
│   ├── parameters.npy
│   ├── aux.npy
│   ├── tau_grid.npy
│   ├── metadata.json
│   └── preview_curves.png
│
└── checkpoints/              # trained SBI model and training metadata
    └── sbi_npe_run/
        ├── posterior.pkl
        ├── inference.pkl
        ├── density_estimator_state_dict.pt
        ├── split_indices.npz
        ├── tau_grid.npy
        ├── run_config.json
        ├── training_summary.json
        ├── training_summary.pkl
        └── loss_curve.png
```

Evaluation products are generated locally by `evaluate_sbi.py` and are not included in the repository.

---

## 1. Generate the synthetic magnetic-field data set

The forward model evolves the normalized magnetic field

$$
b \equiv \frac{B}{B_0},
\qquad
\tau \equiv \Omega_{i,0} t.
$$

The four independent physical coordinates are

$$
\theta =
\left(
|\Delta_0|,
\left|\frac{\delta\Delta}{\Delta_0}\right|,
\frac{\Gamma_d}{\Gamma_c^{(\mathrm{inst})}},
R_{\mathrm{coll}}
\right),
$$

where

$R_{\mathrm{coll}}=\frac{\nu_{ii}}{\nu_{\mathrm{scatt},0}}.$

All four coordinates are sampled uniformly in $\log_{10}$.

| Parameter | Sampling range ||---|---:|
| $|\Delta_0|$ | $10^{-4}$ -- $10^{-1}$ |
| $|\delta\Delta/\Delta_0|$ | $10^{-6}$ -- $10^{-1}$ |
| $\Gamma_d/\Gamma_c^{(\mathrm{inst})}$ | $10^{-4}$ -- $10^{2}$ |
| $R_{\mathrm{coll}}$ | $10^{-2}$ -- $0.9$ |

The default calculation adopts

$$
\beta_0 = 2\times10^{22},
$$

and stores 512 logarithmically spaced time samples over

$$
1 \leq \tau \leq 10^6.
$$

The quasi-stability saturation field is

$$b_{\mathrm{sat}}=\left(\frac{\beta_0 |\Delta_0|}{2}\right)^{1/2}.$$

Only physically amplifying, finite, and approximately monotonic magnetic-field histories are retained.

### Run

```bash
bash make_B_dataset.sh
```

The 100,000-curve data set is stored in:

```text
bfield_dataset_100k/
```

### Data products

`B_curves.npy`
: Normalized magnetic-field histories $B/B_0$, with shape `(N, 512)`.

`parameters.npy`
: Physical parameters with columns

```text
0  |Delta_0|
1  |deltaDelta / Delta_0|
2  Gamma_d / Gamma_c^(inst)
3  nu_ii / Omega_i,0
```

`aux.npy`
: Auxiliary quantities with columns

```text
0  R_coll = nu_ii / nu_scatt,0
1  nu_scatt,0 / Omega_i,0
2  Gamma_c^(inst) / Omega_i,0
3  Gamma_d / Omega_i,0
4  tau_sat
5  B_sat / B0
6  initial db/dtau
```

`tau_grid.npy`
: Dimensionless time grid.

`metadata.json`
: Forward-model configuration, parameter ranges, and data-set metadata.

`preview_curves.png`
: Preview of a subset of the generated magnetic-field histories.

---

## 2. Train the SBI model

The amortized neural posterior estimator is trained in logarithmic coordinates,

$
\theta_{\log} =
\left(
\log_{10}|\Delta_0|,
\log_{10}\left|\frac{\delta\Delta}{\Delta_0}\right|,
\log_{10}\frac{\Gamma_d}{\Gamma_c^{(\mathrm{inst})}},
\log_{10}R_{\mathrm{coll}}
\right).
$

The magnetic-field input is transformed as

$x(\tau)=\frac{\log_{10}[B(\tau)/B_0]}{10.5}.$

The inference model uses:

- a causal convolutional neural network for time-series embedding;
- a neural spline flow (NSF) conditional density estimator;
- neural posterior estimation (NPE) implemented with `sbi`.

The default training configuration is:

```text
Causal CNN layers       : 4
Pooling kernel          : 4
Embedding dimension     : 64

NSF hidden features     : 128
NSF transforms          : 8
Spline bins             : 10

Batch size              : 4096
Learning rate           : 2e-4
Validation fraction     : 0.10
Maximum epochs          : 400
```

For the 100,000-curve data set, 10,000 curves are reserved as a completely held-out test set. The remaining 90,000 curves form the development sample used for training and validation.

### Run

```bash
bash train_sbi_npe.sh
```

Training outputs are stored in:

```text
checkpoints/sbi_npe_run/
```

Important checkpoint products include:

```text
posterior.pkl
inference.pkl
density_estimator_state_dict.pt
split_indices.npz
run_config.json
training_summary.json
loss_curve.png
```

The held-out test curves are not passed to SBI training.

---

## 3. Evaluate the trained posterior

Evaluate the trained posterior on a held-out magnetic-field history with:

```bash
bash evaluate_sbi.sh
```

The evaluation script draws posterior samples for the selected held-out case and propagates posterior samples through the forward model to construct posterior predictive magnetic-field histories.

A typical evaluation uses:

```text
Posterior samples       : 20,000
Posterior predictive    : 200 curves
```

These values can be changed through the command-line arguments of `evaluate_sbi.py`.

The evaluation can generate:

```text
observation_curve.png
marginal_posteriors_log.png
corner_posterior_log.png
posterior_predictive.png
posterior_summary.json
posterior_samples_log.npy
posterior_samples_physical.npy
posterior_predictive_curves.npy
true_theta_log.npy
true_theta_physical.npy
```

These products are generated locally and are not tracked in this repository.

---

## Software requirements

The main Python dependencies are:

```text
numpy
scipy
matplotlib
torch
sbi
```

A CUDA-enabled PyTorch installation is recommended for SBI training. The forward-model data generation and posterior-predictive integrations can also be performed on CPU.

Because the appropriate PyTorch installation depends on the local CUDA environment, install the PyTorch build suitable for your system before installing `sbi`.

---

## Reproducibility

- Synthetic parameter sampling uses a fixed random seed unless changed through command-line options.
- The development/test split is stored in `split_indices.npz`.
- The held-out test sample is not used during NPE training.
- `metadata.json` records the data-generation configuration.
- `run_config.json` records the main SBI architecture and training settings.
- Large evaluation products are intentionally excluded from the repository.

---

## Scientific interpretation

The goal of the SBI analysis is to determine which plasma parameters are robustly encoded in the magnetic-field history and which remain degenerate.

In the adopted forward model, the equilibrium pressure anisotropy $|\Delta_0|$ directly controls the saturation amplitude through

$
b_{\mathrm{sat}} \propto |\Delta_0|^{1/2},
$

whereas the perturbation amplitude and damping ratio primarily affect the transient amplification history. Posterior predictive reconstruction and parameter identifiability should therefore be interpreted separately: an accurately reconstructed $B(t)$ history does not necessarily imply unique recovery of every underlying plasma parameter.

---

## Citation

If you use this code in scientific work, please cite the associated paper:

> J.-H. Ha, *Neural Simulation-Based Inference of Plasma Parameters from Cosmic Magnetic-Field Evolution*.

Publication information and DOI will be added when available.

---

## Contact

Ji-Hoon Ha

Korea Astronomy and Space Science Institute (KASI)

E-mail: jhha@kasi.re.kr
