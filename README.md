# 📧 Spam Email Classification using Machine Learning

A machine learning-based **Spam Email Classification** system that automatically classifies emails as **Spam** or **Not Spam** using their subject and body content.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for classification.

---

##  Project Overview

Spam and unwanted emails are a common problem in modern email systems. Traditional keyword-based filtering can sometimes incorrectly classify emails.

This project focuses on building a machine learning model that learns patterns from previously labeled emails and predicts whether a new email is:

* 🟢 **Not Spam**
* 🔴 **Spam**

The current version focuses on the machine learning classification pipeline. A future version will extend this into a complete web-based email security system with features such as confidence-based classification, sender trust scoring, user feedback, and false-positive prevention.

---

##  Features

### Current Features

* 📂 Load email dataset from CSV
* 🧹 Handle missing values
* 📩 Combine email **Subject + Body**
* 🔤 Convert text into numerical features using **TF-IDF**
* 🤖 Train a **Logistic Regression** classifier
* 📊 Evaluate training accuracy
* 🧪 Evaluate test accuracy
* 📋 Generate classification report
* 🔲 Generate confusion matrix
* ✉️ Classify a new/unseen email as Spam or Not Spam

---

##  Machine Learning Pipeline

```text
                  Email Dataset
                       │
                       ▼
              Handle Missing Values
                       │
                       ▼
              Subject + Email Body
                       │
                       ▼
                 Train / Test Split
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
           Training           Testing
              │                 │
              ▼                 │
        TF-IDF Vectorizer       │
              │                 │
              ▼                 │
       Logistic Regression      │
              │                 │
              └────────┬────────┘
                       ▼
                   Prediction
                       │
              ┌────────┴────────┐
              ▼                 ▼
           🟢 NOT SPAM       🔴 SPAM
```

---

## 🔍 How It Works

### 1. Dataset Loading

The email dataset is loaded from:

```text
spam_email_dataset.csv
```

The dataset contains email information including:

* Subject
* Body
* Spam Label

---

### 2. Data Preprocessing

Missing values are replaced with empty strings to prevent errors during text processing.

```python
maildata = data.where(pd.notnull(data), '')
```

The email subject and body are combined:

```python
X = maildata['Subject'].astype(str) + ' ' + maildata['Body'].astype(str)
```

The target variable is:

```python
Y = maildata['Spam Label']
```

Where:

```text
0 → Not Spam
1 → Spam
```

---

### 3. Train-Test Split

The dataset is divided into training and testing sets.

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)
```

Approximately:

```text
80% → Training
20% → Testing
```

---

## 🔤 TF-IDF Feature Extraction

Machine learning algorithms cannot directly understand raw email text.

Therefore, **TF-IDF** is used to convert the email text into numerical features.

```python
TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
```

TF-IDF gives higher importance to words that are useful for distinguishing between different emails.

For example, words commonly associated with spam may include:

```text
free
winner
prize
offer
claim
urgent
congratulations
```

---

## 🤖 Machine Learning Model

The project uses:

### Logistic Regression

```python
model = LogisticRegression()

model.fit(Xtrain, Y_train)
```

Logistic Regression is suitable for binary classification problems such as:

```text
Spam       → 1
Not Spam   → 0
```

After training, the model can predict the category of previously unseen emails.

---

## 📊 Model Evaluation

The model is evaluated using:

### Accuracy

```python
accuracy_score(Y_test, predictions_test)
```

### Confusion Matrix

```python
confusion_matrix(Y_test, predictions_test)
```

### Classification Report

```python
classification_report(
    Y_test,
    predictions_test
)
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

These metrics provide more information than accuracy alone.

---

## ✉️ Example Prediction

The system can classify a new email such as:

```text
Congratulations! You've won a free ticket to Bahamas.
Click here to claim your prize.
```

The email is converted into TF-IDF features and passed to the trained model.

Example output:

```text
The email is classified as SPAM.
```

---

## 🛠️ Technologies Used

| Technology          | Purpose                 |
| ------------------- | ----------------------- |
| Python              | Programming language    |
| Pandas              | Dataset handling        |
| NumPy               | Numerical operations    |
| Scikit-learn        | Machine learning        |
| TF-IDF              | Text feature extraction |
| Logistic Regression | Email classification    |

---

## 📁 Project Structure

```text
Spam-Email-Classification/
│
├── spam_email_dataset.csv
├── spam_classifier.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Spam-Email-Classification.git
```

### 2. Navigate to the project

```bash
cd Spam-Email-Classification
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

Or use:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Make sure the dataset is present in the project directory.

Then run:

```bash
python spam_classifier.py
```

The program will:

1. Load the dataset
2. Preprocess the emails
3. Split the dataset
4. Generate TF-IDF features
5. Train the Logistic Regression model
6. Calculate training and testing accuracy
7. Display the confusion matrix
8. Display the classification report
9. Predict whether a new email is Spam or Not Spam

---

## 📈 Current Results

The current implementation achieved:

```text
Training Accuracy: 1.0
Test Accuracy:     1.0
```

> Note: Model performance can vary depending on the dataset and train/test split. Accuracy alone is not sufficient to determine the quality of a spam classifier, so precision, recall, F1-score, and the confusion matrix are also evaluated.

---

## Current Limitation

The current system primarily uses the **text content of an email** for classification.

Real-world email systems can face a significant problem called a **false positive**:

```text
Actual Email:     NOT SPAM
Model Prediction: SPAM
```

This can cause legitimate emails such as:

* Job opportunities
* College notifications
* Interview invitations
* Bank notifications
* Important personal emails

to be incorrectly moved into the spam folder.

Reducing these false positives is the main direction planned for the next version of this project.

---

# Future Enhancements

The project is planned to evolve into a complete **AI-powered Email Security System**.

### 1. Confidence-Based Classification

Instead of only:

```text
SPAM / NOT SPAM
```

the system will provide a risk score:

```text
Spam Probability: 25%
→ Safe

Spam Probability: 55%
→ Needs Review

Spam Probability: 92%
→ High Risk
```

This can prevent uncertain emails from being automatically moved to spam.

---

### 2. False Positive Protection

Introduce an additional layer that considers:

* Sender history
* Trusted contacts
* Domain information
* Previous user interaction
* Email content
* Suspicious links
* Attachments

This will help reduce legitimate emails being incorrectly classified as spam.

---

### 3. Sender Trust Score

Each sender can receive a dynamic trust score:

```text
Sender Trust Score: 92/100
```

The score can consider previous interactions and user feedback.

---

### 4. 🔗 URL and Link Analysis

Analyze links inside emails for suspicious characteristics such as:

* Unknown domains
* URL shorteners
* Suspicious URL patterns
* Excessive redirects
* Potential phishing indicators

---

### 5. 📎 Attachment Risk Analysis

Analyze email attachments and identify potentially suspicious file types or characteristics.

---

### 6. Explainable AI

Instead of simply displaying:

```text
SPAM
```

the system will explain why the email was classified as suspicious.

Example:

```text
Spam Risk: 82%

Reasons:
🔴 Suspicious URL
🔴 Unknown sender
🟠 Urgent language
🟠 Promotional content
```

---

### 7. 🔄 User Feedback

Users will be able to correct predictions:

```text
       AI Prediction
            ↓
          SPAM
            ↓
      User says "Not Spam"
            ↓
       Store Feedback
            ↓
       Improve System
```

This will help the system learn from classification mistakes.

---

### 8.  Full-Stack Web Application

The machine learning model will eventually be integrated into a web application.

Planned stack:

```text
Frontend
React + Tailwind CSS
        │
        ▼
Backend
Node.js / Express
        │
        ├──────► MongoDB
        │
        ▼
ML Service
Python + FastAPI
        │
        ▼
Spam Classification Model
```

---

### 9. 📊 Email Security Dashboard

A dashboard will provide:

```text
Total Emails Analyzed
Spam Emails
Safe Emails
Emails Under Review
False Positives
False Negatives
Average Risk Score
```

---

## 🎯 Project Goal

The long-term goal of this project is to move beyond a simple spam classifier and build an **AI-powered email security system that not only detects spam but also reduces false-positive classifications of legitimate emails.**

```text
Current Project

Email
  ↓
TF-IDF
  ↓
Logistic Regression
  ↓
Spam / Not Spam


Future System

Email
  ↓
Content Analysis
  ↓
ML Classification
  ↓
Sender Trust
  ↓
URL Analysis
  ↓
Attachment Analysis
  ↓
Confidence Score
  ↓
User Feedback
  ↓
Safe / Review / Spam
```

---

## Author

**Harini V**

B.Tech Artificial Intelligence & Data Science

Interested in **AI/ML, Full-Stack Development, and Generative AI**.

---

##  Future Vision

> **Building a smarter email security system that protects users from spam without making them miss legitimate emails.**
