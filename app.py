
from flask import Flask, render_template, request, jsonify
import pandas as pd
import urllib.request
import zipfile
import io
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

app = Flask(__name__)

MODEL_FILE = "best_spam_classifier.pkl"
RESULTS_FILE = "model_comparison_results.csv"


def clean_text(message):
    message = str(message).lower()
    message = re.sub(r"https?://\S+|www\.\S+", " URL ", message)
    message = re.sub(r"\S+@\S+", " EMAIL ", message)
    message = re.sub(r"\b\d+\b", " NUMBER ", message)
    message = re.sub(r"\s+", " ", message)
    return message.strip()


def load_dataset():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        filename = next(
            name for name in archive.namelist()
            if name.endswith("SMSSpamCollection")
        )

        df = pd.read_csv(
            archive.open(filename),
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8"
        )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["clean_message"] = df["message"].apply(clean_text)

    return df.reset_index(drop=True)


def train_and_save_model():
    print("Downloading dataset and training models. Please wait...")

    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
            algorithm="brute"
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1,
                random_state=42
            ),
            n_estimators=50,
            random_state=42
        )
    }

    trained_models = {}
    results = []

    for name, classifier in classifiers.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        trained_models[name] = model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, pos_label="spam", zero_division=0
            )
        })

        print(f"Trained: {name}")

    results_df = pd.DataFrame(results).sort_values(
        "F1 Score", ascending=False
    ).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_FILE)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model by spam F1 Score: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved results: {RESULTS_FILE}")

    return best_model


if os.path.exists(MODEL_FILE):
    try:
        best_model = joblib.load(MODEL_FILE)
        print("Loaded saved ML model.")
    except Exception:
        best_model = train_and_save_model()
else:
    best_model = train_and_save_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a valid message."}), 400

    if len(message) > 5000:
        return jsonify({"error": "Message is too long. Maximum 5000 characters."}), 400

    prediction = best_model.predict([clean_text(message)])[0]

    return jsonify({
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
