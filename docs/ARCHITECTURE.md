# Architecture & Technical Design

This document details the technical architecture, design decisions, and file structure of the CropGuard AI project. It serves as a reference for future development and troubleshooting.

## 🔄 Data Flow

The end-to-end pipeline follows this path:
1. **Raw Dataset**: PlantVillage dataset organized by class folders.
2. **Augmentation (`src/augmentation.py`)**: Realism-focused pipeline (albumentations) applying transformations like rotation, flip, brightness/contrast, motion blur, JPEG compression, cutout, and synthetic background paste via GrabCut.
3. **Training (`src/train.py`, `src/data_pipeline.py`)**: Two-phase transfer learning on MobileNetV3-Small. Features balanced class weighting to handle dataset imbalance.
4. **Evaluation (`src/evaluate.py`)**: Assessment of val/test/top-3 accuracy and per-class confusion patterns.
5. **Quantization (`src/quantize.py`)**: Conversion from float32 Keras model to a lightweight, dynamic-range `.tflite` model.
6. **Web Deployment (In Progress)**: LiteRT.js loads the `.tflite` model in the browser via WebAssembly/XNNPACK for entirely offline on-device inference.

## 🧠 Key Design Decisions

* **Scope (3/38 crops)**: The MVP specifically scopes to Tomato, Potato, and Corn diseases (17 classes). This allows us to prove the pipeline's effectiveness before scaling up to all 38 crops in PlantVillage.
* **Two-Phase Transfer Learning**: Initial phase trains only the dense classification head to prevent destroying the pretrained feature extractors. Phase 2 unfreezes the top layers of MobileNetV3-Small for fine-tuning at a lower learning rate.
* **Realism-focused Augmentation**: PlantVillage consists mostly of single leaves on uniform backgrounds. To bridge the domain gap to real phone photos, the pipeline simulates motion blur, JPEG compression, artifacts, and synthetically pastes leaves onto complex backgrounds using GrabCut.
* **Balanced Class Weights**: Handled extreme class imbalance without discarding data. Proved effective via per-class metrics: e.g., Potato_healthy achieved 0.88 precision and 0.96 recall despite having only 152 training images.
* **Dynamic-Range over Full-INT8**: We shipped dynamic-range quantization instead of full INT8. MobileNetV3 is architecturally sensitive to per-tensor INT8 activation quantization due to its hard-swish and Squeeze-and-Excite blocks. Dynamic-range retains float32 activations, allowing the fast XNNPACK delegate to work while heavily compressing weights. *(See `docs/quantization_findings.md` for the deep dive).*

## 📁 File & Module Map

| Module | Description |
|---|---|
| `src/config.py` | Global configuration constants (image size, batch size, paths, class names). |
| `src/data_pipeline.py` | Data loading, dataset splitting, and `tf.data.Dataset` preparation. |
| `src/augmentation.py` | The Albumentations-based realism augmentation pipeline (including GrabCut background synthesis). |
| `src/train.py` | The main two-phase transfer learning routine, metric logging, and model saving. |
| `src/evaluate.py` | Computes test metrics, top-3 accuracy, and analyzes confusion matrices. Identified explainable confusion patterns (e.g., visually similar Tomato brown-lesion diseases and Corn Northern Leaf Blight/Gray leaf spot). |
| `src/quantize.py` | TFLite conversion logic, including the calibration dataset generator attempts. |
| `src/benchmark_pipeline.py` | Utility for timing and profiling inference latencies. |

## ⚠️ Known Environment Quirks

Documented here to save future debugging time:

1. **Windows GPU Limitation**: TensorFlow >= 2.11 does not support native GPU acceleration on Windows. Training requiring heavy compute was offloaded to Google Colab (T4 GPU).
2. **Application Control Policy**: A local security policy on the Windows environment blocks certain native DLLs needed by Python data science libraries.
3. **Albumentations API Changes**: The project uses specific albumentations transformations whose API parameters may have drifted in newer versions. If augmenting locally, ensure the package version strictly matches `requirements.txt`.
4. **Keras 3 vs tfmot Compatibility**: The `tensorflow-model-optimization` package (v0.8.1) is hardcoded to expect legacy Keras 2 types (`tf_keras.src...`). Attempting QAT natively on Keras 3 models will crash.
