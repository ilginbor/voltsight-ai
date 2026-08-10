# Ankara Spatial Cross-Validation Plan

## Configuration

- Grid rows: 102,745
- Spatial block size: 5 km
- Cross-validation folds: 5
- Spatial blocks: 1,157
- Blocks containing at least one positive sample: 24
- Total positive samples: 46

## Fold Distribution

- Fold 0: 20,549 rows, 10 positives, 232 blocks, 5 positive blocks, 0.0487% positive
- Fold 1: 20,549 rows, 9 positives, 231 blocks, 4 positive blocks, 0.0438% positive
- Fold 2: 20,549 rows, 9 positives, 231 blocks, 5 positive blocks, 0.0438% positive
- Fold 3: 20,549 rows, 9 positives, 231 blocks, 5 positive blocks, 0.0438% positive
- Fold 4: 20,549 rows, 9 positives, 232 blocks, 5 positive blocks, 0.0438% positive

## Balance

- Minimum fold rows: 20,549
- Maximum fold rows: 20,549
- Minimum fold positives: 9
- Maximum fold positives: 10

## Assignment Strategy

Spatial blocks are 5 x 5 kilometre groups anchored to the projected
CRS coordinate system.

The assignment algorithm runs in two stages.

First, blocks containing existing charging-station cells are assigned
while prioritizing positive-count balance across folds.

Second, blocks containing no positive samples are assigned while
prioritizing total row-count balance.

A complete block always belongs to exactly one fold.

## Interpretation

This is spatial block cross-validation rather than ordinary random or
stratified row-level splitting.

Cells inside the same 5-kilometre block are kept together. This reduces
local train-validation dependence compared with random splitting.

However, cells located on opposite sides of neighboring block
boundaries can still belong to different folds. Therefore this method
should not be described as eliminating every possible form of spatial
dependence.

## Rare-Class Limitation

Only 46 of 102,745 grid cells
contain an existing charging station.

Model evaluation should therefore emphasize:

- average precision / PR-AUC
- recall
- precision
- F1
- ROC-AUC as a secondary metric
- fold-level stability

Accuracy should not be used as the primary metric.

## Outputs

- `data/processed/ankara_spatial_cv_folds.csv`
- `data/processed/ankara_spatial_cv_block_summary.csv`

## Generated At

2026-08-10T08:45:02.098975+00:00
