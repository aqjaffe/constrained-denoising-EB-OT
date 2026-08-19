# Constrained Denoising, Empirical Bayes, and Optimal Transport

This repository reproduces the figures and tables in the paper

> **Constrained Denoising, Empirical Bayes, and Optimal Transport**
> Adam Quinn Jaffe, Nikolaos Ignatiadis, and Bodhisattva Sen.
> [arXiv:2506.09986](https://arxiv.org/abs/2506.09986)

## Overview

In the denoising problem, Bayes and empirical Bayes (EB) methods can "overshrink" their
output relative to the latent variables of interest. This work studies **constrained
denoising** problems that mitigate such shrinkage. At the oracle level (latent distribution
known), optimal transport is used to characterize the solutions of (i) variance-constrained,
(ii) distribution-constrained, and (iii) general-constrained denoising problems. At the
empirical level, the paper gives a modular procedure that turns any suitable unconstrained
EB denoiser into a constrained one, with explicit rates of convergence.

Each notebook in this repository is self-contained and regenerates one figure or table from
the paper; the pre-computed outputs are stored under [`figs/`](figs/).

## Repository structure

```
.
├── app_astronomy.ipynb      # Astronomy application (stellar chemical abundances)
├── app_baseball.ipynb       # Baseball application (rookie batting skill)
├── app_marketing.ipynb      # Marketing application (e-mail effect)
├── sim_1d.ipynb             # 1-D simulation (denoising & deconvolution errors)
├── sim_2d.ipynb             # 2-D simulation (error table + denoiser scatter panels)
├── sim_heterosked.ipynb     # Heteroskedastic simulation (marginal vs. conditional VCB)
├── npeb/                    # Nonparametric EB package (NPMLE mixture models)
│   ├── GLMixture.py         #   Gaussian-location mixtures
│   ├── PMixture.py          #   Poisson mixtures
│   └── NCHGMixture.py       #   Noncentral hypergeometric mixtures
├── data/                    # Datasets and the notebooks used to build them
│   ├── make_astronomy_dataset.ipynb
│   ├── make_baseball_dataset.ipynb
│   └── ...                  #   raw .csv / .pkl files
├── figs/                    # Pre-computed figure outputs (.pdf and .png)
└── requirements.txt
```

## Environment and installation

The notebooks were developed with **Python 3.9.13**. The key dependencies are pinned in
[`requirements.txt`](requirements.txt):

| Package | Version |
|---|---|
| POT (imported as `ot`) | 0.9.5 |
| CVXPY | 1.3.2 |
| Mosek | 11.0.5 |

`numpy`, `pandas`, `scipy`, and `matplotlib` are also required (any recent release compatible
with Python 3.9 should work).

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Mosek license.** The convex programs (NPMLE and the distribution-/general-constrained
> optimal-transport problems) are solved with the Mosek interior-point solver, which requires
> a license. A free academic license is available at
> <https://www.mosek.com/products/academic-licenses/>.

## Running the notebooks

Each notebook can be run top to bottom independently:

```bash
jupyter lab        # or: jupyter notebook
```

Running a notebook regenerates the corresponding files in [`figs/`](figs/).

### Approximate runtimes

The times below are wall-clock times measured on a personal laptop (a 2020 MacBook Pro) and
are provided only as a rough guide; the two-dimensional simulation and the astronomy
application are the most expensive.

| Notebook | Approx. runtime |
|---|---|
| `app_marketing.ipynb` | < 15 seconds |
| `sim_heterosked.ipynb` | < 30 seconds |
| `app_baseball.ipynb` | ~1 minute |
| `sim_1d.ipynb` | ~15 minutes |
| `app_astronomy.ipynb` | ~18 minutes |
| `sim_2d.ipynb` | ~40 minutes |

## Figure and table mapping

The table below maps each artifact in this repository to its location in the paper. The
repository side is filled in; **the paper-reference column should be completed against the
current version of the manuscript.**

| Repository artifact | Produced by | Description | Location in paper |
|---|---|---|---|
| `figs/marketing.{pdf,png}` | `app_marketing.ipynb` | Marketing application: denoised distribution of the e-mail effect | _TODO_ |
| `figs/sim_1d.{pdf,png}` | `sim_1d.ipynb` | 1-D simulation: denoising & deconvolution error curves vs. sigma^2 and tau^2 | _TODO_ |
| `figs/sim_2d.{pdf,png}` | `sim_2d.ipynb` | 2-D simulation: oracle and empirical denoiser scatter panels | _TODO_ |
| _(LaTeX table printed inline)_ | `sim_2d.ipynb` | 2-D simulation: Monte Carlo denoising & deconvolution errors | _TODO_ |
| `figs/astronomy.{pdf,png}` | `app_astronomy.ipynb` | Astronomy application: denoised stellar chemical abundances | _TODO_ |
| `figs/baseball.{pdf,png}` | `app_baseball.ipynb` | Baseball application: denoised rookie batting skill | _TODO_ |
| `figs/sim_heterosked.{pdf,png}` | `sim_heterosked.ipynb` | Heteroskedastic simulation: marginal vs. conditional variance constraints | _TODO_ |

## Data

- **Astronomy.** Oxygen–Iron and Nitrogen–Iron relative abundances for red-clump stars from
  the Apache Point Observatory Galactic Evolution Experiment (APOGEE) survey. The raw catalog
  is preprocessed in `data/make_astronomy_dataset.ipynb`.
- **Baseball.** Runs-batted-in and games-played counts for rookie batters (final minor-league
  and first major-league seasons), from publicly available [FanGraphs](https://www.fangraphs.com/)
  data. Preprocessed in `data/make_baseball_dataset.ipynb`.
- **Marketing.** The Hillstrom e-mail marketing dataset (`data/hillstrom.csv`).

## Computational notes

- **G-modeling.** The discrete NPMLE is computed via the [`npeb`](https://github.com/jake-soloff/npeb)
  framework (CVXPY/Mosek over a fixed grid of atoms, followed by expectation–maximization).
  The smooth NPMLE with a lower-bounded component covariance follows Magder and Zeger (1996).
- **Variance constraints.** Solved in closed form using the Bures–Wasserstein geometry; no
  numerical optimization is required beyond the unconstrained EB step.
- **Distribution- and general constraints.** Solved as linear programs over a discretized
  grid for the latent coordinate, using CVXPY with the Mosek solver.

## Citation

```bibtex
@article{jaffe2025constrained,
  title   = {Constrained Denoising, Empirical Bayes, and Optimal Transport},
  author  = {Jaffe, Adam Quinn and Ignatiadis, Nikolaos and Sen, Bodhisattva},
  journal = {arXiv preprint arXiv:2506.09986},
  year    = {2025}
}
```

## License

_To be added._
