# Module 4: Applied Deep Learning with PyTorch ⚡

**From PyTorch Fundamentals to Modern Gen AI**

**📍 Location:** `4_PyTorch/`
**🎯 Prerequisite:** Module 3 – Neural Networks from Scratch
**➡️ Next Module:** MLOps & Deployment (The Grand Finale)

Welcome to **Module 4** of **SAIR** – your comprehensive journey into applied deep learning with PyTorch. This module bridges theory and practice, taking you from tensor operations all the way to state-of-the-art object detection with YOLOv8 and Gen AI with transformers.

---

## 🎯 Module Overview

This module is structured in progressive phases:

1. **PyTorch Fundamentals** – Tensors, autograd, training loops
2. **Computer Vision with CNNs** – From scratch implementations to modern architectures
3. **Production-Ready Demos** – Real-world applications with YOLOv8
4. **Research & Theory** – Foundational papers and modern vision concepts

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![UV](https://img.shields.io/badge/UV-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

</div>

---

## 📚 Module Contents

### **Core Learning Track (Notebooks)**

| File | Focus |
|------|-------|
| **`1_PyTorch_Fundamentals/1_Intro.ipynb`** | Tensors, autograd, model building, training loops |
| **`1_PyTorch_Fundamentals/2_DataLoader.ipynb`** | Dataset classes, DataLoader optimization, performance |
| **`2_Vision_and_CNN/3_CNN.ipynb`** | Convolutional Neural Networks from scratch |
| **`2_Vision_and_CNN/4_Transfer_and_ResNet.ipynb`** | Transfer learning, ResNet, pretrained models |
| **`2_Vision_and_CNN/5A.ipynb`**, **`5B.ipynb`**, **`5C.ipynb`** | Advanced vision concepts and experiments |

---

### **Hands-on Labs**

| Lab | Location | Description |
|-----|----------|-------------|
| **Lab 1** | `1_PyTorch_Fundamentals/labs/lab_1.ipynb` | PyTorch basics & tensor operations |
| **Lab 2** | `1_PyTorch_Fundamentals/labs/lab_2.ipynb` | Training neural networks |
| **Lab 3** | `2_Vision_and_CNN/labs/lab_3.ipynb` | CNN implementation and experiments |
| **Lab 4** | `2_Vision_and_CNN/labs/lab_4.ipynb` | Transfer learning & evaluation |

Student submissions are organized under `lab_assignments/[student_name]/`

---

### **Production Demos with YOLOv8**

The `Demos/` directory contains ready-to-run applications:

| Demo | File | Description |
|------|------|-------------|
| **Live Detection** | `demo_01_live_detection.py`         | Real-time object detection with webcam |
| **Background Removal** | `demo_02_background_removal.py` | Remove backgrounds using segmentation |
| **Pose Estimation** | `demo_03_pose_estimation.py`       | Human pose estimation in real-time |
| **Gesture Control** | `demo_04_gesture_control.py`       | Control applications with hand gestures |
| **Model Comparison** | `demo_05_model_comparison.py`     | Compare YOLOv8n/s/m performance |
| **Batch Processing** | `demo_06_batch_processing.py`     | Process multiple images efficiently |
| **Video Processing** | `demo_07_video_processing.py`     | Object detection on video files |

**Quick Start:**
```bash
cd Demos
uv pip install -r requirements.txt
python run_demos.py  # Interactive menu
# Or run individually: python demo_01_live_detection.py
```

Pre-trained models included:
- `yolov8n.pt` – Nano (fastest)
- `yolov8s.pt` – Small
- `yolov8m.pt` – Medium
- `yolov8n-seg.pt` – Segmentation
- `yolov8n-pose.pt` – Pose estimation

---

### **Datasets**

| Dataset | Location | Purpose |
|---------|----------|---------|
| **CIFAR-10** | `data/cifar-10-batches-py/` | CNN training & experiments |
| **COCO128** | `datasets/coco128/` | YOLO format object detection |
| **Assets** | `assets/` | Test images for demos |

The COCO128 dataset is a mini version of COCO with 128 images, perfect for:
- Learning YOLO data format
- Quick experimentation
- Testing detection pipelines

---

### **Research Papers**

| Paper | File | Key Concepts |
|-------|------|--------------|
| **AlexNet** | `papers/AlexNet_paper.pdf` | First deep CNN for ImageNet |
| **ResNet** | `papers/ResNet_paper.pdf` | Residual connections, deep networks |

---

## 📂 Complete Directory Structure

```
4_PyTorch/
│
├── 1_PyTorch_Fundamentals/
│   ├── 1_Intro.ipynb
│   ├── 2_DataLoader.ipynb
│   └── labs/
│       ├── lab_1.ipynb
│       └── lab_2.ipynb
│
├── 2_Vision_and_CNN/
│   ├── 3_CNN.ipynb
│   ├── 4_Transfer_and_ResNet.ipynb
│   ├── 5A.ipynb, 5B.ipynb, 5C.ipynb
│   └── labs/
│       ├── lab_3.ipynb
│       └── lab_4.ipynb
│
├── Demos/
│   ├── demo_01_live_detection.py
│   ├── demo_02_background_removal.py
│   ├── demo_03_pose_estimation.py
│   ├── demo_04_gesture_control.py
│   ├── demo_05_model_comparison.py
│   ├── demo_06_batch_processing.py
│   ├── demo_07_video_processing.py
│   ├── run_demos.py
│   ├── requirements.txt
│   └── README_DEMOS.md
│
├── datasets/
│   └── coco128/
│       ├── images/
│       ├── labels/
│       └── LICENSE
│
├── data/
│   └── cifar-10-batches-py/
│
├── lab_assignments/
│   └── [student_name]/
│       └── lab_1.ipynb
│
├── papers/
│   ├── AlexNet_paper.pdf
│   └── ResNet_paper.pdf
│
├── assets/
│   └── (test images)
│
├── detection_output/
│   └── (processed detection results)
│
├── yolov8n*.pt (model files)
├── coco128.yaml
└── README.md
```

---

## 🚀 Learning Pathway

### **Phase 1: Foundations** (Week 1 - 2)
1. `1_PyTorch_Fundamentals/1_Intro.ipynb` – Tensor operations & autograd
2. Complete `labs/lab_1.ipynb`
3. `1_PyTorch_Fundamentals/2_DataLoader.ipynb` – Data pipelines
4. Complete `labs/lab_2.ipynb`

### **Phase 2: Computer Vision** (Week 3)
1. `2_Vision_and_CNN/3_CNN.ipynb` – Build CNNs from scratch
2. Complete `labs/lab_3.ipynb`
3. `2_Vision_and_CNN/4_Transfer_and_ResNet.ipynb` – Transfer learning
4. Complete `labs/lab_4.ipynb`

### **Phase 3: Modern Vision** (Week 4)
1. Explore `5A_YOLO.ipynb`, `5B_5B_Segment_Pose.ipynb`, `5C_5C_ViTs_and_Deploy.ipynb` for advanced topics
2. Set up and explore `Demos/` environment: `uv pip install -r requirements.txt`
3. Build your own application with COCO128 dataset

---

## 🎯 Learning Outcomes

After completing this module, you will be able to:

- **Build** neural networks from scratch using PyTorch
- **Design** efficient data pipelines with custom Datasets and DataLoaders
- **Train** CNNs for image classification tasks
- **Apply** transfer learning with pretrained ResNet models
- **Deploy** state-of-the-art YOLOv8 models for:
  - Object detection
  - Instance segmentation
  - Pose estimation
- **Process** images, videos, and live camera feeds
- **Understand** modern vision architectures and research papers

---

## 🔧 Installation & Setup with UV

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# Clone the repository (if not done)
git clone [your-repo-url]
cd SAIR/4_PyTorch

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install torch torchvision torchaudio
uv pip install jupyter matplotlib numpy pandas tqdm

# For YOLOv8 demos
cd Demos
uv pip install -r requirements.txt

# Launch Jupyter
cd ..
jupyter notebook
```

### **UV Commands Cheat Sheet**

| Command | Purpose |
|---------|---------|
| `uv venv` | Create virtual environment |
| `uv pip install <package>` | Install a package |
| `uv pip install -r requirements.txt` | Install from requirements file |
| `uv pip list` | List installed packages |
| `uv pip freeze > requirements.txt` | Generate requirements file |
| `uv pip uninstall <package>` | Remove a package |
| `uv cache clean` | Clean uv cache |

---

## 📝 Notes & Best Practices

- **GPU Usage**: Most notebooks automatically detect CUDA. Use `torch.cuda.is_available()` to check.
- **Data Management**: Large datasets are gitignored. Download CIFAR-10 through the notebooks.
- **Model Files**: Pre-trained YOLO models are included, but can be re-downloaded if needed.
- **Lab Submissions**: Place your completed labs in `lab_assignments/[your_name]/`
- **Experimentation**: Use the `detection_output/` folder for saving results.
- **UV Speed**: UV is significantly faster than pip – enjoy the speed boost! ⚡

---

## 🤝 Contributing

Feel free to:
- Add new demos to the `Demos/` directory
- Improve notebook documentation
- Share your lab solutions
- Suggest new datasets or models

---

## 📚 Additional Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Papers with Code](https://paperswithcode.com/)

> *"From tensors to production – understanding every layer of the stack."*

**Happy Learning with UV! 🚀⚡**