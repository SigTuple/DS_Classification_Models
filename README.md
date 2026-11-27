# RBC QI Training

Configuration-driven training for RBC quality-index experiments. The configured experiment set is:

- `e0_baseline`: full-FOV RGB EfficientNet-B0 baseline (3 channels).
- `e1_sobel`: full-FOV RGB + Sobel gradient-magnitude map (4 channels).
- `e2_canny_edge_map`: full-FOV RGB + Canny edge map (4 channels).
- `e3_downscaled_batch2` and `e4_downscaled_batch4`: RGB on a 2x-downscaled dataset.
- `e5_local_entropy`, `e6_local_contrast`, `e7_lbp`: full-FOV RGB plus one structural fourth channel.
- `e8_full_fov_batch4`: full-FOV RGB with batch size four and learning rate `1e-4`.

Every experiment uses the same trainer. Add an experiment by adding a YAML file in `configs/experiments/`; do not copy Python training scripts.

## Layout

```
data/RBC_QI_DATASET/{train,test}/{class_name}/...  # full-FOV dataset (not committed)
data/RBC_QI_DATASET_DOWNSCALED_2X/{train,test}/{class_name}/...  # E3/E4 dataset (not committed)
data/.feature_cache/                              # temporary fourth-channel maps (not committed)
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

The default runner performs E0, E3, E7, E6, E5, E4, E2, then E8. It stops on a failed experiment to preserve a fair, inspectable comparison. To run one experiment:

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/run_experiments.py --configs /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/configs/experiments/e0_baseline.yaml
```

All experiment YAMLs are included in the default E0–E8 queue and can also be run individually with `--configs`.

To continue the queue after a completed or skipped experiment, use its experiment ID:

```bash
python3 scripts/run_experiments.py --start-from e5_local_entropy
```

To resume an interrupted run from its latest completed-epoch checkpoint, keep the same run directory and use:

```bash
python3 scripts/resume_experiment.py --experiment e6_local_contrast
```

## Feature-map cache

For Canny, entropy, contrast, LBP, and Sobel experiments, the fourth-channel map is prepared once before training and reused in every epoch. The status file reports `PREPARING_CACHE` while this happens. The cache is removed after a successful experiment; if a run stops or fails, its cache remains available for the next attempt.

## Overnight status

```bash
cat /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/runs/training_status.log
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/status.py
```

`training_status.log` is intentionally overwritten whenever a state changes. It therefore always says which run is currently training, succeeded, or failed; completed runs retain their own `runs/<run_id>/status.json` and `summary.json`.

The live status includes a PID and heartbeat timestamp. Running `scripts/status.py` checks whether that PID is still alive; an unexpectedly terminated local process is changed from `TRAINING` to `INTERRUPTED` and can be resumed from `checkpoint_last.pth`.

## Export ONNX

Set `DEFAULT_CHECKPOINT_PATH` and `DEFAULT_CONFIG_PATH` at the top of `scripts/export_onnx.py`, then run:

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/export_onnx.py
```

The exporter reads the selected YAML, so RGB experiments get a 3-channel ONNX input and feature-map experiments get a 4-channel input. Deployment must reproduce the configured preprocessing exactly.

## Add an experiment

Copy an experiment YAML and alter only the intended variables. For a fair ablation, preserve the dataset, classes, seed, and training settings. Supported input features are `rgb`, `rgb_canny`, `rgb_local_entropy`, `rgb_local_contrast`, and `rgb_lbp`.
