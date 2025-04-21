# Spectral Clustering on Resting-State fMRI Data

This project applies spectral clustering to resting-state fMRI data to explore functional connectivity patterns in the brain, particularly in the context of neurodegenerative diseases.

## Overview

- Utilizes the AAL brain atlas for anatomical parcellation.
- Implements spectral clustering to group functionally similar brain regions based on temporal coherence.
- Aims to identify alterations in intrinsic connectivity patterns associated with neurodegenerative conditions.

## Files

- `fmri.py` – Main script for preprocessing, clustering, and visualization.
- `data/` – Directory for storing input fMRI datasets (not included).
- `results/` – Output directory for clustering maps and visualizations.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries:
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `nibabel`
  - `nilearn`
  - `mne`

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Script

```bash
python fmri.py
```

Make sure the fMRI data is placed in the correct directory and is formatted appropriately (e.g., NIfTI format).

## Citation

If you use this work or build upon it, please consider citing the relevant references included in the project.

## License

This project is licensed under the [MIT License](LICENSE).
