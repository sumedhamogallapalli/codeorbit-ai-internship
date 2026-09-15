"""
Task 5: Sentiment Analysis on Text Data
CodeOrbit Tech Artificial Intelligence Internship
Author: B.Tech Final Year Student

This program performs sentiment analysis on review text using:
1. TF-IDF (Term Frequency - Inverse Document Frequency) for feature extraction
2. Logistic Regression for multi-class sentiment classification (Positive, Negative, Neutral)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(file_path="sentiment_data.csv"):
    """
    Loads the sentiment dataset from a CSV file.
    Returns:
        DataFrame: Loaded dataset with 'text' and 'sentiment' columns.
    """
    # Check current working directory first, then script's directory
    if not os.path.exists(file_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def train_sentiment_model(df):
    """
    Prepares text data, transforms features using TF-IDF,
    and trains a Logistic Regression classifier.

    Args:
        df (DataFrame): Dataset containing 'text' and 'sentiment' columns.

    Returns:
        tuple: (vectorizer, model, accuracy, report)
    """
    # 1. Separate features (X) and target labels (y)
    X = df["text"]
    y = df["sentiment"]

    # 2. Split dataset into train (80%) and test (20%) sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Extract unigrams and bigrams using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Train multi-class Logistic Regression classifier
    model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluate accuracy and detailed classification report on test set
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return vectorizer, model, acc, report


def predict_sentiment(text, vectorizer, model):
    """
    Predicts the sentiment category of an input review.

    Args:
        text (str): Input review or sentence.
        vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        model (LogisticRegression): Trained Logistic Regression model.

    Returns:
        tuple: (sentiment_label, confidence_percentage)
    """
    # Input validation: check for empty or whitespace-only strings
    if not text or not isinstance(text, str) or text.strip() == "":
        return "Invalid Input", 0.0

    # Vectorize input using the trained TF-IDF vocabulary
    text_vector = vectorizer.transform([text.strip()])

    # Predict the sentiment class and confidence probability
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    return prediction, confidence


def run_example_tests(vectorizer, model):
    """
    Runs automated benchmark predictions on representative sample reviews.
    """
    example_sentences = [
        "This is a great product, I love it and the quality is amazing!",
        "Terrible service, very bad and worst experience ever.",
        "The delivery was standard and package arrived normal.",
        "Poor battery life, completely broken and useless.",
        "Excellent value, highly recommended and works wonderful!",
        "The device has average performance, neither good nor bad.",
    ]

    print("\n" + "=" * 70)
    print("DEMO: PREDICTIONS ON BENCHMARK TEST REVIEWS")
    print("=" * 70)
    print(f"{'Input Review':<52} | {'Predicted Sentiment':<15}")
    print("-" * 70)

    for sentence in example_sentences:
        sentiment, conf = predict_sentiment(sentence, vectorizer, model)
        print(f"{sentence:<52} | {sentiment} ({conf:.1f}%)")
    print("=" * 70 + "\n")


def interactive_mode(vectorizer, model):
    """
    Runs an interactive console loop allowing users to enter custom reviews.
    """
    print("=" * 70)
    print("INTERACTIVE SENTIMENT CLASSIFIER")
    print("Enter any product/movie review to detect its sentiment.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter review text: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting Sentiment Analysis program. Thank you!")
                break

            if not user_input:
                print("Notice: Input text cannot be empty. Please enter a valid sentence.")
                continue

            sentiment, conf = predict_sentiment(user_input, vectorizer, model)
            print(f"--> Predicted Sentiment : [{sentiment.upper()}] (Confidence: {conf:.1f}%)")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    print("=" * 70)
    print("   CodeOrbit Tech AI Internship - Task 5: Sentiment Analysis   ")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[1/4] Loading local sentiment dataset (sentiment_data.csv)...")
    try:
        df = load_dataset("sentiment_data.csv")
        print(f"      Successfully loaded {len(df)} review samples.")
        print("      Sentiment category breakdown:")
        for cat, count in df["sentiment"].value_counts().items():
            print(f"        - {cat}: {count} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Train Model
    print("\n[2/4] Vectorizing with TF-IDF and training Logistic Regression model...")
    vectorizer, model, accuracy, report = train_sentiment_model(df)
    print("      Model training complete!")
    print(f"      Accuracy on unseen test split: {accuracy * 100:.1f}%\n")
    print("Classification Report:")
    print(report)

    # 3. Benchmark Testing
    print("[3/4] Running automated benchmark test reviews...")
    run_example_tests(vectorizer, model)

    # 4. Interactive Console Mode
    print("[4/4] Starting interactive console mode...")
    interactive_mode(vectorizer, model)


if __name__ == "__main__":
    main()
