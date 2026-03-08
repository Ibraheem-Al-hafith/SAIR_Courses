# 🏗️ Classification Hub — SAIR PyTorch Mastery

> You covered all the theory. You followed the lectures.
> Now here are five real datasets and five real problems.
> Build the solutions yourself.

---

## Exercises

| # | Modality | Project | Dataset |
|---|----------|---------|---------|
| [Ex 1](Ex_1_Tabular_Classification.ipynb) | 📊 Tabular | Rice Type Classifier | [Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) |
| [Ex 2](Ex_2_Image_Classification.ipynb) | 🖼️ Image (scratch) | Animal Face Classifier | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces) |
| [Ex 3](Ex_3_Image_Classification_Pretrained.ipynb) | 🌿 Image (pretrained) | Bean Leaf Disease Detector | [Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification) |
| [Ex 4](Ex_4_Audio_Classification.ipynb) | 🎵 Audio | Quran Reciter Identifier | [Kaggle](https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification) |
| [Ex 5](Ex_5_Text_Classification_Transformers.ipynb) | 📝 Text | Sarcasm Detector | [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) |

---

## What Every Submission Must Include

Every notebook you submit must have all of the following:

- A **trained PyTorch model** that solves the stated problem
- A **training report** — loss and accuracy curves for train and validation
- A **final test accuracy score** on held-out data
- A **live inference demo** — the model runs on a new real example and returns a prediction

For **Ex 3 and Ex 5** you also need a written explanation (in a markdown cell)
of your architectural decisions: what model you chose, what you froze, what you added, and why.

For **Ex 4** you need a written explanation of how you converted audio into
a format your model could process.

---

## Note on Exercise 4

Audio classification was not demonstrated step-by-step in the lectures.
You are expected to research the pipeline independently.
You have the mental model — a CNN that classifies 2D input.
The missing piece is how to get from a `.wav` file to a 2D array.
Figure that out and the rest is familiar territory.

---

## Setup (Google Colab)

All exercises run on **Google Colab with a T4 GPU**.

**Runtime → Change runtime type → T4 GPU**

For Kaggle dataset downloads:
```python
from google.colab import files
files.upload()  # upload your kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

*SAIR PyTorch Mastery Course — Classification Hub*
