# Unsupervised Learning Project: VAE for Music Clustering

Course: Neural Networks  
Prepared By: Moin Mostakim

This repo implements an unsupervised pipeline inspired by Variational Autoencoders (VAE) to learn latent representations from music features and cluster tracks.

## What’s implemented (Easy Task)

- Audio feature extraction (MFCC summary statistics)
- Lyrics-only feature extraction (TF-IDF over lyrics)
- Basic VAE (MLP encoder/decoder) for latent feature extraction
- Clustering on latent codes using K-Means
- Baseline: PCA + K-Means
- Metrics: Silhouette Score, Calinski–Harabasz, Davies–Bouldin (and optional ARI/NMI if labels exist)
- Visualizations: t-SNE or UMAP scatter of latent space

## Data format

Create a CSV manifest at `data/metadata.csv` with columns:

- `id` (unique string)
- `audio_path` (path to audio file, relative to repo root or absolute)
- `language` (optional, e.g., `en`)
- `genre` (optional)
- `label` (optional ground-truth label for ARI/NMI; can be language/genre)
- `lyrics` (optional raw lyric text)

Example row:

```csv
id,audio_path,language,genre,label,lyrics
song001,data/audio/song001.wav,en,pop,en,"some lyrics here"
```

## Setup

```bash
# Option A: quick helper (creates .venv if missing)
INSTALL_DEPS=1 source scripts/activate_env.sh

# Option B: manual
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download JamendoLyrics (English subset)

This project expects a `data/metadata.csv` manifest. You can generate a 1000-song subset using Hugging Face datasets (default) or Kaggle.

The recommended way is the repo-level downloader which fetches the raw MP3/lyrics files from Hugging Face Hub and writes the manifest the pipeline expects.

If access is gated, login once:

```bash
huggingface-cli login
```

Then download the English subset into `data/audio/` and `data/lyrics/`, and write `data/metadata.csv`:

```bash
python3 download.py --language English
```

Notes:

- The downloader avoids `datasets` audio decoding (no `torchcodec` requirement).
- You can change the dataset repo or language:

```bash
python3 download.py --repo-id jamendolyrics/jamendolyrics --language English
```

## Download MTG-Jamendo audio (no lyrics)

If your next task needs audio features, you can also download a subset of the Hugging Face dataset `rkstgr/mtg-jamendo`.

Important: MTG-Jamendo provides **full audio + tags** (genres/instruments/moods), but **does not include lyrics**.

This script downloads `.opus` audio files and generates a compatible manifest CSV:

```bash
python scripts/import_hf_mtg_jamendo_audio.py \
  --split train \
  --max-tracks 500 \
  --data-dir data/jamendo
```

Output:

- `data/jamendo/audio/*.opus`
- `data/jamendo/metadata_mtg_jamendo_train.csv`

## Run: VAE → Latents → Clustering

```bash
python -m src.run_pipeline \
  --metadata data/metadata.csv \
  --feature mfcc \
  --latent-dim 16 \
  --clusters 4 \
  --viz umap \
  --outdir results
```

Convolutional VAE on time-frequency features (MFCC frames or log-mel spectrogram):

```bash
python -m src.run_pipeline \
  --metadata data/jamendo_demo/metadata_mtg_jamendo_train.csv \
  --feature logmelspec \
  --vae-arch conv2d \
  --latent-dim 16 \
  --clusters 4 \
  --cluster-method kmeans \
  --viz umap \
  --outdir results_audio_conv
```

Try different clusterers (K-Means / Agglomerative / DBSCAN):

```bash
python -m src.run_pipeline \
  --metadata data/jamendo_demo/metadata_mtg_jamendo_train.csv \
  --feature logmelspec \
  --vae-arch conv2d \
  --cluster-method dbscan \
  --dbscan-eps 0.8 \
  --dbscan-min-samples 10 \
  --viz umap \
  --outdir results_audio_dbscan
```

Hybrid audio + lyrics embeddings (requires the SAME track IDs to have both audio_path and lyrics):

```bash
python -m src.run_pipeline \
  --metadata data/metadata.csv \
  --feature hybrid \
  --audio-feature mfcc \
  --lyrics-embed-dim 128 \
  --latent-dim 16 \
  --clusters 4 \
  --viz umap \
  --outdir results_hybrid
```

Lyrics-only (no audio required):

```bash
python -m src.run_pipeline \
  --metadata data/metadata_known_categories_preprocessed_balanced.csv \
  --feature lyrics_tfidf \
  --latent-dim 16 \
  --clusters 4 \
  --viz umap \
  --outdir results
```

To generate a t-SNE visualization instead:

```bash
python -m src.run_pipeline \
  --metadata data/metadata.csv \
  --feature mfcc \
  --clusters 4 \
  --viz tsne \
  --outdir results
```

## Run: PCA baseline

```bash
python -m src.run_pipeline \
  --metadata data/metadata.csv \
  --feature mfcc \
  --baseline pca \
  --clusters 4 \
  --viz umap \
  --outdir results
```

PCA baseline with t-SNE plot:

```bash
python -m src.run_pipeline \
  --metadata data/metadata.csv \
  --feature mfcc \
  --baseline pca \
  --clusters 4 \
  --viz tsne \
  --outdir results
```

## Compare baseline (Silhouette + Calinski–Harabasz)

After running both `vae` and `pca` once (they append rows to the same CSV), print the latest comparison:

```bash
python3 scripts/compare_baseline.py
```

## Outputs

- `results/clustering_metrics.csv`
- `results/latent_visualization/*.png`
- `results/latents.npz` (latent vectors + ids)

## Notes

- Audio loading uses `librosa`; supported formats depend on your system codecs.
- If you don’t have `label`, ARI/NMI are skipped.
