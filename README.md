# chrompert

CLI application to predict changes in chromatin accessibility at cis-regulatory elements from ChIP-seq features (ported from [aodgulka/accessibility-prediction-from-chipseq](https://github.com/aodgulka/accessibility-prediction-from-chipseq)).

## Prerequisites

- Python >= 3.9
- `bedtools` installed and available on your `PATH` (used by `feature-analysis` to intersect peak regions)

## Installation

```bash
pip install -e .
```

## Usage

### 1. FRiP and Peak Count Filtering (`frip-filter`)
Downloads ENCODE metadata from an ENCODE file report `.tsv`, filters for the highest FRiP score per ChIP target, and downloads the corresponding `.bed.gz` files:

```bash
chrompert frip-filter -i path/to/encode_report.tsv -m 1000 -p 8
```

### 2. Feature Analysis (`feature-analysis`)
Overlaps differential accessibility peaks with the downloaded ChIP-seq BED files using `bedtools intersect` and trains a Random Forest Classifier (RFC) and Ridge Regression (RR) model:

```bash
chrompert feature-analysis -i path/to/differential_peaks.tsv -fc logfc -d BAFdep -x CTCF -p -j 4
```

### 3. Reset BED & Metadata Directory (`reset-bed`)
Clears all downloaded files in `data/bed`, `data/metadata`, and `data/log`:

```bash
chrompert reset-bed
```
