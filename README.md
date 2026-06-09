This is Bhupender Nayak

# PyTorch Training Pipeline: Diabetes Prediction

This repository contains a clean, end-to-end PyTorch training pipeline for binary classification. The model predicts the onset of diabetes based on diagnostic measures using the classic Pima Indians Diabetes Database.

This project serves as a foundational template for structuring modern PyTorch workflows, moving away from raw tensor manipulation to utilizing PyTorch's optimized `torch.nn` modules.

## 🧠 Key Concepts Demonstrated

* **Custom Neural Networks:** Subclassing `nn.Module` to build a Multi-Layer Perceptron (MLP).
* **Data Handling:** Using `TensorDataset` and `DataLoader` for efficient Mini-Batch Gradient Descent.
* **Numerical Stability:** Implementing `BCEWithLogitsLoss` over standard manual Sigmoid + BCELoss for better mathematical stability during backpropagation.
* **Advanced Optimization:** Utilizing the Adam optimizer (`optim.Adam`) for dynamic learning rate adjustments.
* **Training Loops:** Structuring professional training and evaluation modes (`model.train()` and `model.eval()`).

## 📊 Dataset

The dataset used is the **Pima Indians Diabetes Dataset**.
* **Input Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
* **Target (1):** Outcome (0 = No Diabetes, 1 = Diabetes).

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
