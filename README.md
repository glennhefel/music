# Unsupervised Music Clustering (Audio + Lyrics)

This repo builds a multilingual music dataset (English + Bangla included) and runs an unsupervised clustering pipeline using VAE-based embeddings and baselines.

Key capabilities:
- Features: lyrics TF-IDF, MFCC summary, MFCC frames, log-mel spectrograms, hybrid audio+lyrics, and a simple multimodal variant.
- Models/baselines: VAE (MLP/Conv2D), Beta-VAE via `--beta`, CVAE via `--cond-col`, plus baselines (`pca`, `ae`, `raw`).
- Clustering: KMeans, Agglomerative, DBSCAN, Spectral.
- Metrics: silhouette, Calinski-Harabasz, Davies-Bouldin, plus ARI/NMI/purity when a label column is available.

## Data manifests (already in `data/`)

The pipeline is driven by CSV manifests. The most important ones:
- `data/metadata_audio_lyrics_mixed.csv`: paired audio+lyrics (multilingual).
- `data/metadata_known_categories_preprocessed_balanced.csv`: lyrics-only dataset used for the Easy task (balanced; good for en vs bn clustering).
- `data/metadata.csv`: default manifest written by the downloader.

Expected columns (depending on feature mode):
- Required: `id`
- Audio features: `audio_path`
- Lyrics features: `lyrics` (or `lyrics_clean` if present)
- Labels for evaluation: `label` (or pass `--label-col language`, etc.)
- Optional metadata for plots: `language`, `category`/`genre`

## Setup (Windows / PowerShell)

```powershell
cd E:\JetBrains\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download JamendoLyrics subset (optional)

Downloads audio into `data/audio/`, lyrics into `data/lyrics/`, and writes `data/metadata.csv`:

```powershell
python download.py --language English
```

You can keep all languages:

```powershell
python download.py --language all
```

If Hugging Face access is gated:

```powershell
huggingface-cli login
```

## Run all experiment suites (Easy / Medium / Hard)

Canonical runner:

```powershell
python scripts\run_all_tasks.py --device cpu
```

Compatibility shortcut (forwards to the same runner):

```powershell
python run_all_tasks.py --device cpu
```

Quick smoke run:

```powershell
python scripts\run_all_tasks.py --device cpu --viz none --epochs-easy 1 --epochs-medium 1 --epochs-hard 1
```

Where results go (created next to the runner script):
- `scripts/results_easy_task/`
- `scripts/results_medium_task/`
- `scripts/results_hard_task/`

Each task folder contains per-run subfolders plus an aggregated comparison CSV:
- `easy_task_comparison.csv`
- `medium_task_comparison.csv`
- `hard_task_comparison.csv`

## Run a single pipeline experiment

Lyrics-only TF-IDF example:

```powershell
python -m src.run_pipeline ^
  --metadata data\metadata_known_categories_preprocessed_balanced.csv ^
  --feature lyrics_tfidf ^
  --baseline pca ^
  --clusters 2 ^
  --label-col language ^
  --viz none ^
  --outdir scripts\scratch_run
```

Audio log-mel ConvVAE example:

```powershell
python -m src.run_pipeline ^
  --metadata data\metadata_audio_lyrics_mixed.csv ^
  --feature logmelspec ^
  --vae-arch conv2d ^
  --baseline none ^
  --clusters 6 ^
  --label-col language ^
  --viz tsne ^
  --outdir scripts\scratch_audio
```

## Utility scripts

- `scripts/preprocess_lyrics_mixed.py`: writes a `lyrics_clean` column for cleaner TF-IDF.
- `scripts/compare_baseline.py`: compares the latest `vae` vs `pca` rows inside a metrics CSV (pass `--metrics` if your output folder is not `results/`).
- `scripts/tune_lyrics_vae.py`: optional tuning helper for lyrics VAE.
- `scripts/fetch_bangla_lyrics.py`: optional Bangla lyrics scraper template.
  - Requires extra deps: `pip install requests beautifulsoup4` (and optionally `langid`).

## Outputs

Every pipeline run writes:
- `clustering_metrics.csv` (appended per run)
- `latents.npz` (embeddings + ids)
- `latent_visualization/` (plots when `--viz tsne|umap`)
- `reconstructions/` (when `--save-recon`)

## Report

The project writeup is in `PROJECT_REPORT.md`.
