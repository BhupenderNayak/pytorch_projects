# PyTorch Fundamentals: Architecture, Training & MLOps

This branch contains foundational Machine Learning pipelines built from scratch using PyTorch. It demonstrates the progression from raw tensor manipulation to deploying professional, object-oriented neural network architectures using `torch.nn`.

## 🧠 Core Engineering Concepts Mastered
Throughout these notebooks, the following Deep Learning principles were successfully implemented:
* **Custom Architectures:** Building subclasses of `nn.Module` and utilizing `nn.Sequential` for clean, forward-pass pipelines.
* **The 5-Step Training Loop:** Manual implementation of Gradient Descent (Forward Pass, Loss Calculation, Zero Grad, Backpropagation, Optimizer Step).
* **Data Leakage Prevention:** Strict separation of training and testing phases using `model.eval()` and `torch.inference_mode()`.
* **Model Persistence (MLOps):** Safely exporting and importing optimized model weights using `state_dict` and `pathlib` for crash-proof directory management.
* **Telemetry & Visualization:** Tracking Train vs. Test loss and plotting real-time dimensional data using Scikit-Learn and Matplotlib.

## 📂 Project Files & Milestones

### 1. `DiabetesModel_pytorch.ipynb`
* **Description:** First project and initial milestone for building diagnostic classification models.
* **Tech:** Binary classification using `BCEWithLogitsLoss` and Mini-Batch Gradient Descent via `DataLoader`.

### 2. `Task_1.ipynb`
* **Description:** Designed a custom `TelemetryClassifier` to predict user active/resting states based on 6-dimensional mock sensor data.
* **Tech:** Custom Multi-Layer Perceptron (MLP) architecture mapping 6 input features down to a single probability output.

### 3. `learnpytorch_io_module_1execrise.ipynb`
* **Description:** End-to-end linear regression pipeline serving as the final Chapter 1 foundational exercise. 
* **Tech:** The model successfully learned to replicate a hidden mathematical formula ($y = 0.3x + 0.7$) using pure SGD optimization and Mean Absolute Error (`nn.L1Loss`).

## 🛠️ Tech Stack
* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Data Manipulation:** Pandas, NumPy
* **Data Processing & Visualization:** Scikit-Learn (`train_test_split`, `StandardScaler`, `LabelEncoder`), Matplotlib

---
*Author: Bhupender Nayak*
