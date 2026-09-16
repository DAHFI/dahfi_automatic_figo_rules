# DAHFI Automatic FIGO Rules

This repository provides an automated, rule-based implementation of the
**FIGO (International Federation of Gynecology and Obstetrics) criteria**
for the analysis and classification of **cardiotocographic (CTG) recordings**.

The implemented methodology analyzes fetal heart rate (FHR) and uterine
contraction (UC) signals to automatically evaluate the main CTG features
considered in the FIGO-based classification:

- FHR baseline
- FHR variability
- Decelerations
- Uterine contractions
- Overall CTG classification

The main purpose of this repository is to provide a reproducible implementation
of the methodology described in the associated research work and to facilitate
the analysis of CTG recordings in fetal monitoring research.

By applying the same computational criteria consistently across recordings,
the automatic procedure may help reduce part of the variability associated
with manual CTG interpretation.

> **Note:** This repository implements a computational, FIGO-oriented
> interpretation of CTG criteria. Some clinical definitions require an explicit
> algorithmic formulation before they can be applied automatically to digital
> signals. These implementation details are described in the associated article.


## Data Source

The data used in the example notebook are obtained from the publicly available
**CTU-UHB Intrapartum Cardiotocography Database**, available through
[PhysioNet](https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

The database contains real intrapartum cardiotocographic recordings collected
at the University Hospital in Brno, Czech Republic, together with associated
clinical information.

In this repository, the database is used to demonstrate the preprocessing,
automatic CTG analysis, and evaluation workflow implemented in the project.


## Example of Use

An example of how to use the provided code is available in:

`example_of_use.ipynb`

The notebook demonstrates:

- How to load CTG recordings.
- How to visualize the original FHR and UC signals.
- How to preprocess the CTG signals.
- How to visualize the preprocessed recording.
- How to apply the FIGO-based feature analysis to an individual CTG.
- How to apply the same analysis to a complete collection of CTG recordings.

The notebook is intended to provide a simple and reproducible example of the
workflow implemented in this repository.


## Environment Setup

To facilitate reproducibility, the repository includes an environment
configuration file:

`environment.yml`

This file contains the Conda and pip dependencies required to run the project.


### 1. Create the Conda environment

First, make sure that Anaconda or Miniconda is installed.

From the root directory of the repository, run:

```bash
conda env create -f environment.yml