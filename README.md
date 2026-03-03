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

## 🛠️ Usage Instructions

### 1️⃣ Install Dependencies

> Make sure all required libraries are installed:

bash
pip install -r requirements.txt

### 2️⃣ Download the Dataset

> The dataset is not included in this repository.

Please follow the instructions in the data/README.md file to download it from Kaggle and place the CSV file in the correct directory.

### 3️⃣ Configure the Dataset Path

> Verify that the dataset path inside main.py matches your folder structure.
Example:
df_suicide_data = pd.read_csv("data/Suicide_Detection.csv")

### 4️⃣ Choose Preprocessing Version

> Inside main.py, you can select whether to use stopwords removal:

> Keep the stopwords removal line active → WITHOUT stopwords

> Comment the line → WITH stopwords

> Also adjust the sequence length accordingly:
max_len = 160  # Use 320 when running WITH stopwords

### 5️⃣ Run the Experiment

> Execute the script:
python main.py

> The script will automatically:

  🟢 Train the CNN–BiLSTM model

  🟢 Display training and validation curves

  🟢 Generate the confusion matrix and ROC curve

  🟢 Report Accuracy, Precision, Recall, and F1-score

  🟢 Measure training time and memory usage

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











