# 🧠 Automatic detection of suicidal ideation on social media using a hybrid CNN–BiLSTM architecture

This repository contains the **code** and **instructions** to reproduce the experiments from the article:

> **Automatic detection of suicidal ideation on social media using a hybrid CNN–BiLSTM architecture**
>
> Manuscript ID - IEEE LATAM Submission ID: 10144
>
> Authors: Sadala Nagib Salame Neto (sadalaneto@ufpa.br), Kaique da Silva Pinto (kaique.pinto@itec.ufpa.br), Silvio Tadeu Teles da Silva (silvio.teles@itec.ufpa.br), Míercio Cardoso de Alcântara Neto (miercio@ufpa.br)
>
> Year: 2026

---

## 🚀 Introduction

The goal of this project is to apply **Natural Language Processing (NLP)** techniques and **deep neural networks** to detect social media posts with **indications of suicidal ideation**.  
A hybrid **CNN + BiLSTM** architecture was used, achieving competitive results.

---

## 🧠 Methodological Overview

The experimental pipeline implemented in `main.py` includes:

1. Dataset selection and class balancing  
2. Text preprocessing (cleaning, normalization, optional stopword removal)  
3. Tokenization and sequence padding  
4. CNN–BiLSTM model construction  
5. Training with EarlyStopping and learning rate scheduling  
6. Quantitative evaluation (Accuracy, Precision, Recall, F1-score)  
7. Confusion matrix and ROC curve generation  
8. Computational cost measurement (training time and memory usage)  

---

## 🛠️ Usage Instructions

### 1️⃣ Install Dependencies

Open the project folder in your preferred IDE (VS Code, PyCharm, Spyder).

Then install the required libraries using your environment's package manager (for example, via the integrated terminal or Python interpreter settings);

> Alternatively, you may install the libraries manually through your IDE’s package manager interface.

### 2️⃣ Download the Dataset

The dataset is not included in this repository.

> ⚠️ Please follow the instructions in the data/README.md file to download it from Kaggle and place the CSV file in the correct directory.

### 3️⃣ Configure the Dataset Path

Verify that the dataset path inside main.py matches your folder structure.
> Example:
> - df_suicide_data = pd.read_csv("data/Suicide_Detection.csv")

### 4️⃣ Choose Preprocessing Version

Inside main.py, you can select whether to use stopwords removal:

> Keep the stopwords removal line active → WITHOUT stopwords

> Comment the line → WITH stopwords
> - text = ' '.join(word for word in text.split() if word not in stop_words)

Also adjust the sequence length accordingly:
>  max_len = 160  # Use 320 when running WITH stopwords

### 5️⃣ Run the Experiment

Open main.py in your IDE and execute the file using the IDE’s Run or Execute button.

---

## 📊 Main Results

| Configuration             | Accuracy | F1-score |
|---------------------------|----------|----------|
| With stopwords removal    | 95.60%   | 95.60%   |
| Without stopwords removal | 95.77%   | 95.79%   |

---

## ⚠️ Ethical Disclaimer

This project is **strictly academic**.  
Any use in production must undergo an **ethical review** and involve **mental health professionals**.

---

## 📄 License

This project is licensed under the MIT License.  




















