import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# 1. LOAD THE DATASET

data = pd.read_csv(
    'spam_email_dataset.csv',
    encoding='cp1252'
)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

# 2. HANDLE MISSING VALUES

maildata = data.where(pd.notnull(data), '')

print("\nMissing values handled successfully.")

# 3. CHECK SPAM / NOT-SPAM DISTRIBUTION

print("\nSpam Label Distribution:")
print(maildata['Spam Label'].value_counts())

# 4. CREATE INPUT (X) AND OUTPUT (Y)


# Combine Subject and Body into one text column

X = maildata['Subject'].astype(str) + ' ' + maildata['Body'].astype(str)

Y = maildata['Spam Label']


print("\nTotal number of emails:", X.shape[0])
# 5. SPLIT DATA INTO TRAINING AND TESTING DATA


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)


print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
# 6. TF-IDF FEATURE EXTRACTION


feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)


# Fit TF-IDF only on training data

Xtrain = feature_extraction.fit_transform(X_train)

# Transform test data using the same TF-IDF vocabulary

Xtest = feature_extraction.transform(X_test)


print("\nTF-IDF conversion completed.")

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)

# 7. CONVERT LABELS TO INTEGER


Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# 8. CREATE LOGISTIC REGRESSION MODEL


model = LogisticRegression()

model.fit(Xtrain, Y_train)


print("\nModel training completed successfully!")

# 9. TRAINING DATA PREDICTION

predictions_train = model.predict(Xtrain)

accuracy_train = accuracy_score(
    Y_train,
    predictions_train
)

print("TRAINING RESULTS")

print("Training Accuracy:", accuracy_train)

# 10. TEST DATA PREDICTION
predictions_test = model.predict(Xtest)

accuracy_test = accuracy_score(
    Y_test,
    predictions_test
)

print("TESTING RESULTS")
print("Test Accuracy:", accuracy_test)

# 11. CONFUSION MATRIX

cm = confusion_matrix(
    Y_test,
    predictions_test
)

print("CONFUSION MATRIX")

print(cm)

# 12. CLASSIFICATION REPORT

print("CLASSIFICATION REPORT")

print(
    classification_report(
        Y_test,
        predictions_test,
        target_names=[
            "NOT SPAM",
            "SPAM"
        ]
    )
)
# 13. TEST A NEW EMAIL

input_email = [
    "Congratulations! You've won a free ticket to Bahamas. "
    "Click here to claim your prize."
]


# Convert new email into TF-IDF features

input_email_features = feature_extraction.transform(
    input_email
)


# Predict

input_prediction = model.predict(
    input_email_features
)


# 14. DISPLAY NEW EMAIL RESULT

print("NEW EMAIL PREDICTION")


print("Email:")
print(input_email[0])

print("\nPrediction:")

if input_prediction[0] == 1:
    print("The email is classified as SPAM.")
else:
    print("The email is classified as NOT SPAM.")