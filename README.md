# 🤟 ASL Sign Language Recognition — Fundamentals of AI (BYOP)


---

## 👤 Student Details

| Field              | Details                          |
|--------------------|----------------------------------|
| **Name**           | Mansi Kumari                     |
| **Reg. Number**    | 25BAI11449                       |
| **Branch**         | Artificial Intelligence & ML     |
| **University**     | VIT Bhopal University            |
| **Project Type**   | BYOP — Bring Your Own Project    |
| **Course**         | Fundamentals of AI               |

---

## 📌 Project Overview

This project implements a **Real-Time American Sign Language (ASL) Word Recognition System** capable of autonomously extracting, mathematically normalizing, and dynamically classifying complex ASL vocabulary from any standard webcam feed.

### Core Ecosystem Technologies:
- **Google MediaPipe**: Highly optimized dual-hand anatomical tracking (126 coordinates per frame).
- **Mathematical Scale Invariance**: A proprietary geometry translation algorithm that zeroes the wrist and normalizes coordinates against maximum hand boundaries, making the AI **100% immune** to camera-distance and depth distortion.
- **Continuous Learning GUI**: An asynchronous Tkinter interface that allows users to instantly report false-positives mid-inference, autonomously injecting the custom coordinates back into the `asl_hand_landmarks.csv` database, and silently hot-swapping the active AI brain without dropping a single webcam frame.

### Tri-Backend Classifier System:
1. **Gaussian Naive Bayes (NumPy)**: Built mathematically from scratch, proving the probability of class given coordinates (CO4).
2. **K-Nearest Neighbors (NumPy)**: Evaluates strict Euclidean geometry and Bias-Variance tradeoffs across the dataset vectors (CO5).
3. **Random Forest (Scikit-Learn)**: Processes complex morphological structures using 100 parallel mathematical decision-trees to perfectly filter out dynamic noise (like wrist tilt and mirroring).

---

## 🎯 Course Outcomes Covered

| CO   | Topic                        | Where Implemented                          |
|------|------------------------------|--------------------------------------------|
| CO1  | Intelligent Agents           | `predict.py` — continuous learning & dynamic AI hot-swapping |
| CO4  | Probability & Bayes Theorem  | `classifier.py` → `GaussianNaiveBayes` natively coded |
| CO5  | ML Classification            | `classifier.py` → `KNN` + Bias-Variance Analysis |

---

## 🗂️ Project Structure
```
SIGN LANGUAGE/ASL/
│
├── main.py                ← Central CLI router
├── classifier.py          ← Naive Bayes + KNN + StandardScaler (NumPy Native)
├── train.py               ← Multi-backend Training pipeline (Bayes, KNN, RF)
├── predict.py             ← Live Webcam Inference + Interactive Self-Learning GUI
├── evaluate.py            ← Mathematical Accuracy & Confusion Matrices
├── data_extractor.py      ← Scans WLASL_v0.3 videos, extracts 126 coordinate vectors
├── utils.py               ← Core Math: normalize_landmarks() scale invariance
├── config.py              ← 37-Word Vocabulary categorized by hand taxonomy
│
├── dataset/               ← Raw WLASL .mp4 video files + class JSON mappings
├── asl_hand_landmarks.csv ← Processed 126-feature database
├── reported_errors.csv    ← Self-learning telemetry log
└── requirements.txt       ← Dependencies
```

---

## ⚙️ How It Works (The Pipeline)

1. **Extraction (`data_extractor.py`)**: 
   The pipeline jumps exactly to the `25%`, `50%`, and `75%` marks of the raw `.mp4` dataset videos. This exclusively isolates the "Stroke" of the sign—ignoring the moments human actors raise or drop their hands to maximize Signal-to-Noise ratio.
   
2. **Geometric Normalization (`utils.py`)**: 
   All 126 features are passed to `normalize_landmarks()`. The wrist is forced to `(0, 0, 0)` so the coordinates are strictly relative. The matrix is then divided by the greatest absolute bounding radius—guaranteeing that a hand placed two feet from the camera mathematically evaluates exactly the same as a hand placed six inches away.
   
3. **Training & Inference (`train.py` & `predict.py`)**: 
   The database is split 80/20 and fitted to the user's chosen classifier. During live inference, bounding confidence thresholds verify the detected gesture against the 37-word explicitly coded taxonomy map in `predict.py`.

---

## 🚀 Step-by-Step Setup & Execution

*This guide assumes the evaluator has zero prior context regarding the project structure or ASL dataset requirements.*

### 1. Environment Requirements
- **Python 3.10+**
- A standard webcam
- Sufficient disk space for raw video datasets (if choosing to extract from scratch).

### 2. Clone the Repository & Setup Dependencies
It is highly recommended to isolate the project using a Python virtual environment to avoid dependency conflicts.
```bash
# Clone the base tracking repository
git clone https://github.com/mansi25bai11449-uni/ASL-sign-Language-Recognition
cd ASL

# Create and activate a Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install strictly verified dependencies
pip install -r requirements.txt
```

### 3. Procure Required Offline Assets
Because this system processes geometric landmarks entirely offline (no external API calls):
1. **MediaPipe Task File**: The system requires the base `hand_landmarker.task` file in the root directory. If missing, the script will prompt you, but you can download it directly from Google's standard MediaPipe repository.
2. **Kaggle Custom Dataset**: 
   - Download the pre-filtered **1,000-word ASL Dataset** from my Kaggle link: https://www.kaggle.com/datasets/aryanraj801/videos
   - Extract the downloaded `.zip` file so that all `.mp4` video files live strictly inside the `dataset/videos/` directory.
   - Ensure the definition mapping file `WLASL_v0.3.json` exists in `dataset/`.
3. **Core Feature Database (`asl_hand_landmarks.csv`)**:
   - Extract the compressed `asl_hand_landmarks.zip` directly into the root folder before training any of the models.
   - *(Alternatively, if the ZIP is missing, you can directly download the fully compiled 36MB dataset file from my [Kaggle](https://www.kaggle.com/datasets/aryanraj801/asl-hand-landmarks)).*

### 4. Configuration
You can explicitly control which words the AI learns and detects by editing `config.py`. The `COMMON_WORDS` list dictating the vocabulary is currently configured to roughly 37 morphologically distinct signs for maximum real-world reliability.

---

## 🖥️ Execution Guide

Once the repository is configured, you interact with the entire ecosystem through the `main.py` router script.

### Step 1 — Extract the Math (Only needed if building a new dataset)
If you wish to recalculate the Euclidean models from the raw `.mp4` video files:
```bash
# Evaluates thousands of videos, outputting to asl_hand_landmarks.csv
python main.py --extract
```
Note:
---
Moving to Step 2 requires completing Step 1 , Only If  You are thinking to train the models again

Models are already trained for testing purpose ,  _You can skip to Step 3 for testing_ .

---

### Step 2 — Train the Machine Learning Backends
Please check for asl_hand_landmarks.csv (Refer to Step 1) and for model files (`.npz` and `.joblib`) , If Missing then,
You must build the AI's internal model files (`.npz` and `.joblib`) before running the live webcam. You can compile three separate architectures:
```bash
# Train the Naive Bayes model
python main.py --train --method bayes

# Train the K-Nearest Neighbors model
python main.py --train --method knn

# Train the advanced Random Forest (Decision Tree) model
python main.py --train --method rf
```

### Step 3 — Boot the Real-Time Webcam Inference
To test the live gesture tracking against the models you just trained:
```bash
# Recommend using bayes for probabilistic inference enabling noise-tolerant classification through prior and likelihood estimation
python main.py --predict --method bayes

# Recommend using knn for instance-based adaptive classification leveraging local neighborhood similarity for robust real-time predictions
python main.py --predict --method knn

# Recommend using `rf` for absolute performance immunity against webcam mirroring
python main.py --predict --method rf
```
Note:
---
During live inference, you can press **`Esc`** on your keyboard at any time to safely close the webcam window and exit the interface.


---

## 🧠 Continuous Learning (Self-Correction)

If the AI incorrectly predicts your sign during live prediction:
1. Press the **`R`** key on your keyboard.
2. The webcam explicitly suspends its buffer, and an asynchronous GUI window instantly queues on screen.
3. Select whether the AI misunderstood the Shape or the Sign.
4. Input what your intended word was (e.g., `hello` or `bye`).
5. Upon hitting Submit, the system **silently records your exact physical coordinates**, seamlessly integrates them into `asl_hand_landmarks.csv`, immediately executes `train.py` in a threaded background process, and **hot-swaps the AI's internal memory** in less than 2 seconds without you ever having to restart the script!

---


*MIT License — Open source for academic use.*  
*Submitted conditionally representing Fundamentals of AI — BYOP Project.*
*VIT Bhopal University | Mansi Kumari | 25BAI11449*
