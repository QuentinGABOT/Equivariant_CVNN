# Reconstruction Project

This directory contains the entrypoint for the full CVNN reconstruction experiment.

## Description

Uses the shared `cvnn` package to load data, train a complex-valued autoencoder model, reconstruct full SAR images, and visualize results.

## Quickstart

1. Ensure you are in the project root and inside the Poetry environment:
   ```bash
   poetry install
   poetry shell
   ```

2. Run the reconstruction experiment:
   ```bash
   cd projects/reconstruction
   poetry run python run.py
   ```

3. Results (logs, models, plots) will be written to the `logs/` directory as defined in `configs/config_reconstruction.yaml`.

## Configuration

All settings are stored in `configs/config_reconstruction.yaml`. You can tweak data paths, model parameters, optimizer settings, and logging options there.
