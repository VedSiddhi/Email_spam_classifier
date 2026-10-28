import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("spam_dataset.csv")

print("Dataset loaded successfully!")
print("Total messages:", len(data))


# -----------------------------------
# 2. Check Dataset
# -----------------------------------

print("\nDataset columns:")
print(data.columns)

print("\nFirst 5 records:")
print(data.head())


# -----------------------------------
# 3. Separate Input and Output
# -----------------------------------

X = data["message_content"]
y = data["is_spam"]


# -----------------------------------
# 4. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# -----------------------------------
# 5. Convert Text into Numbers
# -----------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# -----------------------------------
# 6. Train Naive Bayes Model
# -----------------------------------

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

print("\nNaive Bayes model trained successfully!")


# -----------------------------------
# 7. Make Predictions
# -----------------------------------

y_pred = model.predict(X_test_vectorized)


# -----------------------------------
# 8. Evaluate Model
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 9. Save Model and Vectorizer
# -----------------------------------

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n-----------------------------------")
print("Files saved successfully!")
print("-----------------------------------")
print("model.pkl")
print("vectorizer.pkl")