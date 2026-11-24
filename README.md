# RBC QI Training

Configuration-driven training for RBC quality-index experiments. The initial experiments are:

- `e0_baseline`: RGB EfficientNet-B0 (3 channels).
- `e2_sobel`: RGB + Sobel-gradient EfficientNet-B0 (4 channels).

Every experiment uses the same trainer. Add an experiment by adding a YAML file in `configs/experiments/`; do not copy Python training scripts.

## Layout

```
data/RBC_QI_DATASET/{train,test}/{class_name}/...  # dataset (not committed)
configs/base.yaml                                  # shared settings
configs/experiments/*.yaml                         # only experiment differences
runs/<run_id>/                                    # checkpoints, metrics, config, ONNX
runs/training_status.log                           # one-line live status for terminal checks
```

## First-time setup

Put the existing dataset in `data/RBC_QI_DATASET/` so its layout is:

```
data/RBC_QI_DATASET/train/coverslip/...
data/RBC_QI_DATASET/test/coverslip/...
```

Install the Python packages in the training environment:

```bash
python3 -m pip install -r requirements.txt
```

## Run the initial comparison

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/run_experiments.py
```

The runner performs E0, then E2. It stops on a failed experiment to preserve a fair, inspectable comparison. To run one experiment:

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/run_experiments.py --configs /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/configs/experiments/e0_baseline.yaml
```

## Overnight status

```bash
cat /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/runs/training_status.log
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/status.py
```

`training_status.log` is intentionally overwritten whenever a state changes. It therefore always says which run is currently training, succeeded, or failed; completed runs retain their own `runs/<run_id>/status.json` and `summary.json`.

## Export ONNX

Set `DEFAULT_CHECKPOINT_PATH` and `DEFAULT_CONFIG_PATH` at the top of `scripts/export_onnx.py`, then run:

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/export_onnx.py
```

The exporter reads the selected YAML, so E0 gets a 3-channel ONNX input and E2 gets a 4-channel input. Deployment must reproduce the configured preprocessing; E2 specifically needs the same Sobel channel generation.

## Add an experiment

Copy an experiment YAML and alter only the intended variables. For a fair ablation, preserve the dataset, classes, seed, and training settings. Supported initial input features are `rgb` and `rgb_sobel`.
