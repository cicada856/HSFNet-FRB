# HSFNet-FRB

HSFNet-FRB is a hierarchical specialist-fusion network for five-class fast radio burst (FRB) morphology classification. A ResNet50 base classifier handles the global A/B/C/C1/C2 task, while a B/C1/C2 specialist and two binary specialists refine the main minority-class confusion boundaries.

On the fixed 268-sample CHIME paper-test split, the selected system obtains **63.43% accuracy**, **53.35% macro-F1**, and **64.03% weighted-F1**, with C1 and C2 recall of **30.00%** and **50.00%**, respectively. The fusion vote bonus (`beta=0.20`) was selected using development-only out-of-fold predictions.

## Repository contents

```text
configs/                 Final reported hyperparameters
data/chime/              Processed CHIME arrays, labels, and event order
data/simulated/          Expected simulation-data layout (data not bundled)
data/splits/             Fixed development/paper-test membership
results/                 Selected test metrics and development-only beta sweep
scripts/                 Reproduction command wrappers
src/hsfnet/              Core training, fusion, evaluation, and audit code
```

Model weights, the approximately 12 GB simulation corpus, paper sources, figures, and supplementary experiment outputs are intentionally excluded.

## Environment

The reported experiments used Python 3.8 and TensorFlow 2.9.

```bash
conda create -n hsfnet-frb python=3.8 -y
conda activate hsfnet-frb
pip install -r requirements.txt
```

## Data

`data/chime/chime_interp_frbs_data.npz` contains 535 processed `256 x 256` CHIME waterfall arrays stored as `uint8`. Each sample was independently normalized by its positive maximum, matching the preprocessing in `src/hsfnet/data.py`. The fixed class-wise half split contains 267 development and 268 paper-test samples.

The full simulation corpus is not included because of its size. Place Frabjous-style simulation files under `data/simulated/simfrb/` as described in [data/simulated/README.md](data/simulated/README.md).

The CHIME-derived data retain the ordering and morphology labels used by this study. Before redistributing a fork, verify that your use complies with the original CHIME/FRB data terms and cite the source publications listed below.

## Reproduction

After installing the simulation data, train all four components with:

```bash
bash scripts/train_final_system.sh
```

Evaluate the trained hierarchy on the fixed paper-test split with:

```bash
bash scripts/evaluate_final_system.sh
```

Individual scripts expose additional options through `--help`. The exact selected settings are recorded in [configs/final_configuration.yaml](configs/final_configuration.yaml).

The fusion parameter can be independently recomputed from the retained out-of-fold predictions:

```bash
python src/hsfnet/experiments/select_vote_bonus_from_oof.py \
  --oof-csv results/validation_beta_selection/oof_component_predictions.csv \
  --output-dir artifacts/beta_selection_check
```

## Data provenance

- CHIME/FRB Collaboration, *The First CHIME/FRB Fast Radio Burst Catalog*, ApJS 257, 59 (2021), https://doi.org/10.3847/1538-4365/ac33ab
- Pleunis et al., *Fast Radio Burst Morphology in the First CHIME/FRB Catalog*, ApJ 923, 1 (2021), https://doi.org/10.3847/1538-4357/ac33ac
- Kumar, Mahabal, and Tendulkar, *Frabjous: Deep Learning Fast Radio Burst Morphologies*, PASA 43 (2026), https://doi.org/10.1017/pasa.2026.10191

## License

The source code is released under the MIT License. Third-party and derived datasets remain subject to their original terms.
