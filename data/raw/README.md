# Raw Dataset

This directory contains the raw AI-generated code used by VibeGuard.

## Structure

- `training/` — file/snippet-level generated samples used to construct
  the training corpus.
- `metadata/` — metadata describing each generated sample.

Each generated code file is identified by a unique `sample_id`.

The raw dataset should not be modified after generation. Cleaning and
processing should produce separate files under `data/processed/`.