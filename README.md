# 🌿 CropGuard AI

**Offline-first plant disease diagnosis for mobile devices.**

CropGuard AI is designed to address connectivity gaps in agricultural regions by putting a highly accurate, lightweight disease classification model directly in the browser. No internet connection is required for inference once the web app is loaded. The MVP is scoped to 17 disease classes across **Tomato, Potato, and Corn**, trained on the PlantVillage dataset.

## 📊 Key Results

| Metric | Value | Notes |
|---|---|---|
| **Test Accuracy** | **0.9465** | Float32 model on 15% stratified test split |
| **Top-3 Accuracy** | **0.9961** | High reliability for top suggestions |
| **Production Accuracy** | **0.9401** | Dynamic-range quantized TFLite model |
| **Model Size** | **1.15 MB** | 8.8× reduction from 10.12 MB float32 baseline |
| **Latency (Median)** | **22.0 ms** | XNNPACK-accelerated on laptop CPU |

## 🏗️ Architecture Overview

The system pipeline revolves around two main components:
1. **Python Training Pipeline**: Handles data ingestion, realism-focused augmentation, two-phase transfer learning on MobileNetV3-Small, and evaluation.
2. **Web Frontend (In Progress)**: A PWA utilizing LiteRT.js to run the `.tflite` model directly on the user's device.

*(See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a detailed technical breakdown of the data flow, design decisions, and modules.)*

## 🔍 Engineering Highlights: The Quantization Debugging Story

Optimizing the model for web deployment required strict size reduction, but initial attempts at full INT8 Post-Training Quantization (PTQ) caused a massive accuracy collapse (0.9465 → 0.7630). 

* **Hypothesis 1 (Class Imbalance)**: We suspected the default uniform random calibration over the highly imbalanced training set starved minority classes (e.g., Potato_healthy only had ~1 representative image). We rewrote the calibration pipeline to use stratified sampling (30 images per class). The result was even worse (0.6836), decisively disproving the imbalance hypothesis.
* **Hypothesis 2 (Architectural Sensitivity)**: Testing a weights-only (dynamic-range) quantized model yielded 0.9401 accuracy. This confirmed the collapse was solely due to *activation* quantization. MobileNetV3's hard-swish activations and Squeeze-and-Excitation (SE) blocks produce narrow, non-linear activation distributions that per-tensor INT8 quantization clips severely.
* **Attempting QAT**: We investigated Quantization-Aware Training (QAT) to recover the lost accuracy, but hit a hard compatibility block: `tensorflow-model-optimization 0.8.1` is tied to legacy Keras 2, while this project was built in Keras 3. 
* **The Solution**: We abandoned full INT8 and shipped the **dynamic-range** model. It sacrifices a negligible 0.64 accuracy points, reduces the model to a web-friendly 1.15MB, and crucially, keeps activations in float32. This allows the model to cleanly leverage the fast **XNNPACK** delegate on the web, yielding a median latency of 22ms.

## 💻 Tech Stack

- **Model**: TensorFlow 2.15+, Keras 3 (MobileNetV3-Small)
- **Data & Augmentation**: Pandas, Albumentations, OpenCV
- **Web Inference**: LiteRT.js (WebAssembly / XNNPACK)
- **Frontend**: Vanilla HTML/JS, CSS (PWA)

## 🚀 Setup and Run

### Environment Setup
1. Clone the repository and create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
> **Note for Windows Users**: TF>=2.11 does not support native GPU on Windows. Training was done on Google Colab (T4 GPU) to bypass this and local Application Control policies that blocked certain native DLLs. Inference and data pipelines work fine locally.

### Running the Web App (Dev)
```bash
cd app
npm install
npm run dev
```

## 📋 Project Status

* **Model Training & Evaluation**: ✅ DONE
* **Quantization & Optimization**: ✅ DONE
* **PWA Frontend (LiteRT.js)**: 🔄 IN PROGRESS
* **Treatment Lookup Table**: ❌ NOT STARTED
* **Real-world Phone-Photo Test Set**: ❌ NOT STARTED

---

## 🖼️ Demo
> **[TODO]**: Insert screenshots and a demo GIF of the PWA once completed.

## 📱 Real-World Testing
> **[TODO]**: Insert results from testing against real-world smartphone photos in varied lighting conditions once completed.

## 💊 Treatment & Actionable Advice
> **[TODO]**: Describe the treatment lookup feature mapping predictions to actionable agricultural advice once built.
