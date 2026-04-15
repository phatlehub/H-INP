# Bridging Intrinsic and Prior Normality for Logical Anomaly Detection in Industrial Visual Inspection

This repository is the code release for our paper and is built on the official INP-Former codebase, which serves as the primary baseline.

The baseline INP-Former training and evaluation scripts are preserved as-is. The new contribution in this repository is an inference-time prior prototype bank and hybrid fusion path for logical anomaly detection. No encoder retraining is required.

## What Is Included

- The original INP-Former scripts for single-class, multi-class, few-shot, zero-shot, and super-multi-class settings.
- Offline prior prototype bank construction from training features.
- Prior-only and hybrid inference modes for ablation and evaluation.
- A lightweight config file and shell wrappers for the new inference path.

## Repository Layout

```text
INP_Former_Single_Class.py
INP_Former_Multi_Class.py
INP_Former_Few_Shot.py
INP_Former_Zero_Shot.py
INP_Former_Super_Multi_Class.py
build_prior.py
test.py
fusion.py
configs/hybrid_config.yaml
prototypes/
models/
scripts/
```

## Environment Setup

```bash
conda create -n INP python=3.8.12 -y
conda activate INP
pip install -r requirements.txt
```

Optional packages:

```bash
pip install gradio
pip install onnx==1.15.0 onnxruntime-gpu==1.15.0 onnxsim
```

## Datasets

Supported datasets:

- MVTec-AD
- VisA in 1-class split format
- Real-IAD

The code accepts an explicit `--data_path`, so datasets can be stored anywhere.

Typical folder structures:

```text
mvtec_anomaly_detection/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    ground_truth/broken_large/*.png

VisA_pytorch/1cls/
  candle/
    train/good/*
    test/good/*
    test/bad/*
    ground_truth/bad/*

Real-IAD/
  realiad_1024/
  realiad_jsons/
```

## Baseline INP-Former

The original INP-Former scripts remain available for training and testing.

Single-class:

```bash
python INP_Former_Single_Class.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --phase train
python INP_Former_Single_Class.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --phase test
```

Multi-class:

```bash
python INP_Former_Multi_Class.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --phase train
python INP_Former_Multi_Class.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --phase test
```

Few-shot:

```bash
python INP_Former_Few_Shot.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --shot 4 --phase train
python INP_Former_Few_Shot.py --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection --shot 4 --phase test
```

Zero-shot:

```bash
python INP_Former_Zero_Shot.py --source_dataset Real-IAD --dataset MVTec-AD --data_path /path/to/mvtec_anomaly_detection
```

Super-multi-class:

```bash
python INP_Former_Super_Multi_Class.py \
  --mvtec_data_path /path/to/mvtec_anomaly_detection \
  --visa_data_path /path/to/VisA_pytorch/1cls \
  --real_iad_data_path /path/to/Real-IAD \
  --phase train

python INP_Former_Super_Multi_Class.py \
  --mvtec_data_path /path/to/mvtec_anomaly_detection \
  --visa_data_path /path/to/VisA_pytorch/1cls \
  --real_iad_data_path /path/to/Real-IAD \
  --phase test
```

## Hybrid Prototype Fusion

### 1. Build the prior prototype bank

Build global normal prototypes from training features with K-Means:

```bash
python build_prior.py \
  --dataset MVTec-AD \
  --data_path /path/to/mvtec_anomaly_detection \
  --encoder dinov2reg_vit_base_14 \
  --K 50 \
  --output_path checkpoints/prior_bank.pt
```

Shell wrapper:

```bash
bash scripts/build_prior.sh
```

### 2. Run evaluation and ablations

Supported modes:

- `inp`: baseline INP-only inference
- `prior`: prior-only inference
- `hybrid`: prior and INP fusion

Using the default hybrid config:

```bash
python test.py \
  --config configs/hybrid_config.yaml \
  --dataset MVTec-AD \
  --data_path /path/to/mvtec_anomaly_detection \
  --save_dir ./saved_results \
  --save_name INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6
```

Override the mode and prior path directly:

```bash
python test.py \
  --dataset MVTec-AD \
  --data_path /path/to/mvtec_anomaly_detection \
  --save_dir ./saved_results \
  --save_name INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6 \
  --mode hybrid \
  --prior_path checkpoints/prior_bank.pt \
  --alpha_mode adaptive
```

Shell wrapper:

```bash
bash scripts/test_hybrid.sh
```

## Config

Default hybrid settings live in `configs/hybrid_config.yaml`:

```yaml
mode: hybrid
K: 50
alpha_mode: adaptive
alpha_fixed: 0.5
prior_path: checkpoints/prior_bank.pt
log_alpha: true
```

## Outputs

Evaluation reuses the baseline metrics:

- Image AUROC
- Image AP
- Image F1-max
- Pixel AUROC
- Pixel AP
- Pixel F1-max
- AUPRO

When `--model_path` is not given, `test.py` expects checkpoints at:

```text
{save_dir}/{save_name}/{item}/model.pth
```

If you want to evaluate a single file, pass `--model_path` explicitly.

Alpha logs, when enabled, are saved as:

```text
{save_dir}/{save_name}/alpha_values_{mode}.npy
```

## Experimental Setup and Benchmark Statistics

### Benchmarks

- MVTec-AD: 15 categories (10 objects + 5 textures), 5,354 training images, 1,725 test images, with pixel-level masks.
- VisA: 12 categories in single-class protocol, 10,821 normal images and 1,200 anomalous images.

### Evaluation Metrics

- Image-level AUROC (I-AUROC)
- Pixel-level AUROC (P-AUROC)
- AUPRO

### Default System Configuration

- Encoder: DINOv2-Base with registers (ViT-B/14, frozen)
- Resize/Crop: 448/392
- Intrinsic prototypes: INP_num = 6
- Prior prototype bank size: K = 50
- Hybrid weighting: adaptive alpha with clipping range [0.2, 0.8]

## Main Quantitative Results

Single-class setting, averaged over all categories.

| Method | MVTec I-AUROC | MVTec P-AUROC | MVTec AUPRO | VisA I-AUROC | VisA P-AUROC | VisA AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PaDiM | 97.9 | 94.4 | 88.2 | 86.2 | 90.1 | 83.4 |
| PatchCore | 99.1 | 97.1 | 93.5 | 90.6 | 98.1 | 85.3 |
| DRAEM | 98.0 | 95.5 | 90.3 | 86.5 | 91.2 | 84.0 |
| SimpleNet | 99.6 | 97.2 | - | 88.4 | 97.9 | - |
| UniFormaly | 99.2 | 96.7 | 93.9 | 88.9 | 97.3 | 84.5 |
| INP-Former (baseline) | 99.7 | 97.8 | 95.1 | 92.4 | 98.6 | 87.2 |
| H-INP (ours) | 99.8 | 98.2 | 95.8 | 93.1 | 98.9 | 88.1 |

Compared with INP-Former baseline, the hybrid method improves:

- MVTec-AD: +0.4 P-AUROC, +0.7 AUPRO
- VisA: +0.3 P-AUROC, +0.9 AUPRO

## Logical Anomaly-Focused Results

Logical-defect subset: transistor, cable, capsule.

| Category | INP-Former P-AUROC | INP-Former AUPRO | H-INP P-AUROC | H-INP AUPRO |
| --- | ---: | ---: | ---: | ---: |
| Transistor | 96.3 | 87.1 | 98.1 | 90.4 |
| Cable | 97.5 | 88.8 | 99.0 | 91.2 |
| Capsule | 98.2 | 90.5 | 99.5 | 92.6 |
| Mean | 97.3 | 88.8 | 98.9 | 91.4 |

Average gain on logical-defect subset:

- +1.5 P-AUROC
- +2.1 AUPRO

These gains are larger than the overall category average, consistent with the motivation of the method.

## Ablations

### Component Ablation

| Variant | All P-AUROC | All AUPRO | Logical P-AUROC | Logical AUPRO |
| --- | ---: | ---: | ---: | ---: |
| INP-only baseline | 97.8 | 95.1 | 97.3 | 88.8 |
| Prior-only (PPB only) | 97.4 | 94.3 | 98.1 | 90.0 |
| Fixed alpha = 0.5 | 97.9 | 95.4 | 98.5 | 90.6 |
| Adaptive alpha (H-INP) | 98.2 | 95.8 | 98.9 | 91.4 |

### Prior Bank Size Sensitivity

P-AUROC on MVTec-AD vs. K:

| K | 10 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P-AUROC | 97.7 | 98.0 | 98.2 | 98.2 | 98.2 |

K = 50 is used as the default in this repository.

### Additional Observation from the Paper

Decoder-level fusion gives a small extra gain (+0.1 P-AUROC) with slightly higher latency, so distance-level fusion is kept as the default deployment path.

## Notes

- Prior and hybrid modes require `--prior_path`.
- The baseline INP-Former scripts and checkpoints can be reused directly for this repository.
- For historical context and the original upstream release, see `README_OLD.md`.

## Acknowledgements

This work builds on INP-Former and is inspired by prior anomaly detection research, including Dinomaly, ADer, Reg-AD, OneNIP, and AdaCLIP.

We acknowledge Innovation FabLab, Ho Chi Minh University of Technology (HCMUT), VNU-HCM for supporting this study.
This work is supported by the Department of Industrial Management, National Taiwan University of Science and Technology, Taipei, Taiwan.

## Citation

If this repository helps your research, please cite the paper
Bridging Intrinsic and Prior Normality for Logical Anomaly Detection in Industrial Visual Inspection.

The BibTeX entry will be added after publication.
