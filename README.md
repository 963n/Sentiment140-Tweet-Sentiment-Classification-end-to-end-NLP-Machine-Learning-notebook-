# Sentiment140-Tweet-Sentiment-Classification-end-to-end-NLP-Machine-Learning-notebook-
# 💬 Sentiment140 Tweet Sentiment Classification
An **end-to-end NLP Machine Learning notebook** (Jupyter/Colab) that classifies tweets into **Negative / Positive / Neutral** using the **Sentiment140** dataset.  
Classic features (**Bag of Words**, **TF-IDF**) + modern **LLM embeddings (Qwen3-Embedding)** — with clear evaluation plots and metrics. ✅

---

## 🔗 Quick Access

[![🧠 Embeddings](https://img.shields.io/badge/Qwen_Embeddings-.npy-6A5ACD?style=for-the-badge)](./qwen_embeddings.npy)
[![📂 Repository](https://img.shields.io/badge/View_on-GitHub-black?style=for-the-badge&logo=github)](https://github.com/963n/Sentiment140-Tweet-Sentiment-Classification-end-to-end-NLP-Machine-Learning-notebook-/edit/main/README.md)


---

## 📸 Preview

Add a screenshot/gif after you run the notebook (recommended):

![Project Preview](./assets/preview.png)

---

## 🧠 How It Works

### 1) Dataset (Sentiment140)

The project uses **Sentiment140** tweets, commonly labeled as:
- `0` → **Negative**
- `4` → **Positive** (converted to `1` in the notebook)
- `2` → **Neutral** (exists in the test set)

The notebook merges train + test files, then **samples 500,000 tweets** for faster experimentation.

**Files used in the notebook:**
- `training.1600000.processed.noemoticon.csv`
- `testdata.manual.2009.06.14.csv`

---

### 2) Preprocessing

Tweets are cleaned and normalized using:
- URL removal
- Mention removal (`@user`)
- Non-letter symbol removal
- Lowercasing
- Stopword removal (**NLTK**)
- Lemmatization (**WordNetLemmatizer**)

---

### 3) Feature Engineering (Two Worlds)

#### A) Classic ML Features
- **Bag of Words** (`CountVectorizer`) with `max_features=5000`
- **TF-IDF** (`TfidfVectorizer`) with `max_features=5000`

#### B) LLM Embeddings (Qwen)
Uses:
- `sentence-transformers`
- Model: **`Qwen/Qwen3-Embedding-0.6B`**
- Optional GPU acceleration via PyTorch CUDA

Embeddings are generated and saved for reuse:
- Saved to: **`qwen_embeddings.npy`**

---

### 4) Models & Evaluation

The notebook trains and evaluates multiple classifiers (depending on feature type), including:
- **Multinomial Naive Bayes** (great baseline for sparse text features)
- **Linear SVM (LinearSVC)**
- **Logistic Regression**

Evaluation includes:
- **Accuracy**
- **Macro Precision / Recall / F1**
- **Confusion Matrix**
- **ROC Curves + Macro ROC-AUC (OVR)**

---

## ✨ Features

- Full pipeline: **load → clean → vectorize → train → evaluate**
- Compares **BoW vs TF-IDF vs LLM embeddings**
- GPU-ready embedding generation
- Saves embeddings to avoid re-computation
- Clear plots (confusion matrices + ROC curves)

---

## 🛠️ Tech Stack

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=fff)](#)
[![pandas](https://img.shields.io/badge/pandas-Data%20Frames-150458?logo=pandas&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy&logoColor=fff)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=fff)](#)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-2E7D32?logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Ready-EE4C2C?logo=pytorch&logoColor=fff)](#)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-6A5ACD?logoColor=fff)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C?logoColor=fff)](#)

---

## 📂 Project Structure

```txt
├── ML_PROJECT (1).ipynb
├── qwen_embeddings.npy              # generated (optional)
├── assets/
│   └── preview.png                  # add your screenshot here (optional)
├── requirements.txt                 # recommended
└── README.md
````

---

## ⚙️ Run Locally

### 1) Install Requirements

```bash
pip install -r requirements.txt
```


### 2) Download Dataset

Download Sentiment140 CSVs (or Kaggle version) and place them in the project folder, then make sure the notebook paths match your files.

### 3) Run Notebook

```bash
jupyter notebook
```

Open: `ML_PROJECT (1).ipynb`

---

## 🔍 Notes & Tips

* **Sampling:** The notebook samples **500k tweets** to speed up training.
* **Neutral class:** Neutral (`2`) mainly appears from the manual test file — ensure your merged dataset contains it if you want 3-class classification.
* **GPU helps a lot** for Qwen embeddings. If CUDA isn’t available, it will run on CPU (slower).
* **Reuse embeddings:** Once saved, load `qwen_embeddings.npy` to skip re-encoding.

---

## 🚀 Future Improvements

* Add **training checkpoints** and clean modular code (turn notebook into scripts)
* Track experiments (Weights & Biases / MLflow)
* Add **calibration** or confidence display for predictions
* Try stronger classifiers (LinearSVM tuning, LightGBM, shallow neural nets)
* Deploy as a **Streamlit app** (tweet input → prediction + confidence + explanation)
* Add explainability (top words for BoW/TF-IDF, embedding-based analysis)

---

## 📚 References

* Sentiment140 Dataset (Kaggle): [https://www.kaggle.com/datasets/kazanova/sentiment140/](https://www.kaggle.com/datasets/kazanova/sentiment140/)
* scikit-learn Docs: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
* NLTK Docs: [https://www.nltk.org/](https://www.nltk.org/)
* Sentence-Transformers Docs: [https://www.sbert.net/](https://www.sbert.net/)
* Qwen Embedding Model (HuggingFace): [https://huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

---

## 🧑‍💻 Author

**Mohammed Alawfi**
📍 Madinah, Saudi Arabia

[![GitHub](https://img.shields.io/badge/GitHub-Profile-000?logo=github\&logoColor=fff)](https://github.com/963n)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin\&logoColor=fff)](https://www.linkedin.com/in/mohammed-a-3913a5378/)


---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

```
```
## 📌 Note About the Training Data (File Size)

The full **Sentiment140 training dataset** is **too large to be included / loaded directly from this repository** (or may fail to upload due to size limits).  
If you need the exact training file(s) used in this project, feel free to **contact me on LinkedIn** and I’ll share them with you.
---
