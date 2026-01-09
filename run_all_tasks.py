"""Compatibility shim.

This repository's canonical runner is scripts/run_all_tasks.py.

This file exists so older commands like `python run_all_tasks.py` still work,
but it forwards execution to scripts/run_all_tasks.py (which writes results
next to itself inside scripts/).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    runner = repo_root / "run_all_tasks.py"
    if not runner.exists():
        raise SystemExit(f"Expected runner not found: {runner}")

    cmd = [sys.executable, str(runner), *sys.argv[1:]]
    print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd)
    return int(completed.returncode)
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--beta",
            "4",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "betavae_logmel_kmeans"),
        ]
    )

    # CVAE conditioned on language
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--cond-col",
            "language",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "cvae_logmel_lang_kmeans"),
        ]
    )

    # Multi-modal: audio + lyrics + genre/category
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "multimodal",
            "--audio-feature",
            "logmelspec",
            "--genre-col",
            "category",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "multimodal_logmel_kmeans"),
        ]
    )

    # Baselines: PCA + KMeans, AE + KMeans, raw + KMeans, raw + Spectral
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "pca",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_pca_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "ae",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_ae_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_raw_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "spectral",
            "--spectral-n-neighbors",
            "10",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language,category",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_spectral_raw_logmel"),
        ]
    )

    aggregate_metrics(results_hard / "hard_task_runs", results_hard / "hard_task_comparison.csv")

    print("\nDone.")
    print("Easy:", results_easy)
    print("Medium:", results_medium)
    print("Hard:", results_hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
