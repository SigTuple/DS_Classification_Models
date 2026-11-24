# RBC QI Training

Configuration-driven training for RBC quality-index experiments. The configured experiment set is:

- `e1_baseline`: RGB EfficientNet-B0 (3 channels).
- `e3_canny_edge_map`: RGB + Canny map (4 channels).
- `e4_downscaled_batch2` and `e5_downscaled_batch4`: RGB on a 2x-downscaled dataset.
- `e6_local_entropy`, `e7_local_contrast`, `e8_lbp`: RGB plus one structural fourth channel.
- `e9_full_fov_batch4`: full-FOV RGB with batch size four.

Every experiment uses the same trainer. Add an experiment by adding a YAML file in `configs/experiments/`; do not copy Python training scripts.

## Layout

```
data/RBC_QI_DATASET/{train,test}/{class_name}/...  # full-FOV dataset (not committed)
data/RBC_QI_DATASET_DOWNSCALED_2X/{train,test}/{class_name}/...  # E4/E5 dataset (not committed)
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

The runner performs E1, then E3 through E9. It stops on a failed experiment to preserve a fair, inspectable comparison. To run one experiment:

```bash
python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/run_experiments.py --configs /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/configs/experiments/e1_baseline.yaml
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

The exporter reads the selected YAML, so RGB experiments get a 3-channel ONNX input and feature-map experiments get a 4-channel input. Deployment must reproduce the configured preprocessing exactly.

## Add an experiment

Copy an experiment YAML and alter only the intended variables. For a fair ablation, preserve the dataset, classes, seed, and training settings. Supported input features are `rgb`, `rgb_canny`, `rgb_local_entropy`, `rgb_local_contrast`, and `rgb_lbp`.
