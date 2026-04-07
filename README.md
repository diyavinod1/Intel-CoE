# 🌿 Plant Classification using Intel oneAPI 🚀

> ⚡ AI-powered plant species classification using Deep Learning + Intel oneAPI optimization

---

## 🧠 Overview

This project uses **Deep Learning** to classify plant images into:

* 🌱 **Purple Chloris**
* 🌿 **Crowfoot Grass**
* 🌸 **Celosia Argentea L**

Built using **Transfer Learning (MobileNetV2)** and optimized for performance using **Intel oneAPI (IPEX)**.

---

## ✨ Features

* 🔥 High Accuracy (>95%)
* 🧠 Transfer Learning (MobileNetV2)
* ⚡ Intel oneAPI Optimization (IPEX)
* 📊 Clean Training + Evaluation Pipeline
* 🖼 Image Prediction Support
* 📁 Organized Project Structure

---

## 🏗 Project Structure

```
plant-classifier/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best_model.pth
│
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* 🐍 Python
* 🔥 PyTorch
* 🧠 MobileNetV2 (Pretrained)
* ⚡ Intel oneAPI (Intel Extension for PyTorch - IPEX)

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant-classifier.git
cd plant-classifier
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Dataset Setup

Make sure your dataset is structured as:

```
dataset/
 ├── train/
 ├── val/
 ├── test/
```

---

### 5️⃣ Train the Model

```bash
python train.py
```

---

### 6️⃣ Evaluate Model

```bash
python evaluate.py
```

---

### 7️⃣ Predict on New Image

```bash
python predict.py
```

---

## ⚡ Intel oneAPI Integration

This project leverages **Intel oneAPI** via **Intel Extension for PyTorch (IPEX)** for optimized CPU performance.

* ✅ Automatically enabled if supported
* ⚠️ Falls back gracefully if not available (e.g., macOS)

---

## 📊 Results

| Metric        | Value                           |
| ------------- | ------------------------------- |
| Accuracy      | ~95%+ 🔥                        |
| Model         | MobileNetV2                     |
| Training Type | Transfer Learning + Fine-Tuning |

---

## 🧪 Sample Output

```
Prediction: CROWFOOT_GRASS
Confidence: 97.34%
```

---

## ⚠️ Notes

* Intel oneAPI optimization works best on **Linux/Windows with Intel CPUs**
* On macOS, the project runs without IPEX but maintains performance

---

## 💡 Future Improvements

* 🌐 Web UI (Streamlit)
* 📱 Real-time camera detection
* 📊 Confusion matrix visualization
* ☁️ Deployment (Cloud / API)

---

## 🤝 Contributing

Feel free to fork this repo and improve it 🚀

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

## 👨‍💻 Author

Made with 💻 + ☕ by **Diya Vinod**

---
