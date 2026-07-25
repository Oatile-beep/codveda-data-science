"""
Codveda Data Science Internship - Level 3, Task 2: NLP - Text Classification
Author: Thapelo Oatile Tlhomelang

Goal: Classify text data into categories (spam vs. non-spam).

Note on the dataset: a synthetic but realistic set of spam and legitimate
("ham") messages is generated locally from templates and word banks, so this
runs fully offline. This also avoids depending on NLTK's downloadable corpora
(punkt, stopwords), which require an internet connection on first use and can
fail depending on your network - the same lesson learned from the Task 1
web scraper needing a live site. Stemming still uses NLTK's PorterStemmer,
which is a pure algorithm bundled with the library and needs no download.

Steps performed:
1. Preprocess text (tokenization via regex, stopword removal, stemming).
2. Convert text into numerical representation using TF-IDF.
3. Train classification models (Naive Bayes and Logistic Regression).
4. Evaluate using precision, recall, and F1-score.
"""

import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

RANDOM_STATE = 42
stemmer = PorterStemmer()


def generate_spam_ham_dataset(random_state=RANDOM_STATE):
    """Generate a synthetic but realistic labelled dataset of spam and
    legitimate ("ham") messages using templates and word banks, so the task
    runs fully offline with no external dataset download required."""
    rng = np.random.default_rng(random_state)

    spam_templates = [
        "CONGRATULATIONS! You have WON a {prize}! Click {link} now to claim your prize!!!",
        "URGENT: Your account will be suspended. Verify now at {link} to avoid closure.",
        "Get {prize} FREE! Limited time offer, click {link} before it's gone!",
        "You've been selected for a {prize} giveaway! Claim at {link} immediately!",
        "Act now! Exclusive {prize} deal just for you, visit {link} today only!",
        "WINNER! Your number was picked for a {prize}. Reply YES or visit {link}.",
        "Cheap loans available NOW! Apply at {link}, no credit check needed!",
        "Hot singles in your area want to meet you! Click {link} to see photos!",
        "Your PayPal account needs verification. Click {link} immediately or lose access!",
        "Make ${amount} a week working from home! No experience needed, visit {link}!",
    ]
    ham_templates = [
        "Hey, are we still on for {event} this {day}?",
        "Can you send me the {doc} before {day}? Thanks!",
        "Reminder: {event} has been moved to {day}.",
        "Thanks for the help with the {doc} earlier, really appreciate it.",
        "Let's grab coffee sometime this {day} if you're free.",
        "The {event} meeting notes are attached, let me know your thoughts.",
        "Happy birthday! Hope you have a great {day}.",
        "Just checking in - how did the {event} go?",
        "Could you review the {doc} when you get a chance?",
        "See you at {event} on {day}, looking forward to it.",
    ]

    prizes = ["iPhone", "cash prize", "vacation", "gift card", "laptop", "lottery jackpot"]
    links = ["bit.ly/xyz123", "claim-now.net", "free-prize.com", "verify-account.info"]
    amounts = ["500", "1000", "2500", "5000"]
    events = ["the meeting", "lunch", "the project review", "the workshop", "the call"]
    days = ["Monday", "Tuesday", "Friday", "the weekend", "tomorrow"]
    docs = ["report", "spreadsheet", "presentation", "assignment", "invoice"]

    # A handful of genuinely ambiguous/borderline templates, mixing
    # "spammy-sounding" words into legitimate messages and vice versa, so the
    # classification task isn't trivially easy - closer to real-world text.
    borderline_ham_templates = [
        "URGENT: can you send the {doc} today? Deadline moved up, please reply ASAP.",
        "Reminder - click the link in the calendar invite to join {event}.",
        "Free coffee in the break room today, come grab some before {event}!",
        "Limited spots left for {event} - confirm by {day} if you're attending.",
    ]
    borderline_spam_templates = [
        "Hi, following up on the {doc} - can you review it by {day}? Thanks.",
        "Reminder about {event}, see you there on {day}.",
    ]

    messages, labels = [], []

    for _ in range(130):
        template = rng.choice(spam_templates)
        msg = template.format(
            prize=rng.choice(prizes), link=rng.choice(links), amount=rng.choice(amounts)
        )
        messages.append(msg)
        labels.append("spam")

    for _ in range(20):
        template = rng.choice(borderline_spam_templates)
        msg = template.format(doc=rng.choice(docs), day=rng.choice(days), event=rng.choice(events))
        messages.append(msg)
        labels.append("spam")

    for _ in range(130):
        template = rng.choice(ham_templates)
        msg = template.format(
            event=rng.choice(events), day=rng.choice(days), doc=rng.choice(docs)
        )
        messages.append(msg)
        labels.append("ham")

    for _ in range(20):
        template = rng.choice(borderline_ham_templates)
        msg = template.format(doc=rng.choice(docs), day=rng.choice(days), event=rng.choice(events))
        messages.append(msg)
        labels.append("ham")

    df = pd.DataFrame({"message": messages, "label": labels})
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def preprocess_text(text):
    """Lowercase, tokenize with a simple regex (avoids needing NLTK's
    downloadable 'punkt' tokenizer), remove stopwords (using scikit-learn's
    built-in list, avoiding NLTK's downloadable 'stopwords' corpus), and
    stem each remaining token."""
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam")
    recall = recall_score(y_test, predictions, pos_label="spam")
    f1 = f1_score(y_test, predictions, pos_label="spam")

    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (spam class)")
    print(f"  Recall:    {recall:.4f} (spam class)")
    print(f"  F1-score:  {f1:.4f} (spam class)")

    return predictions, {"model": name, "accuracy": accuracy, "precision": precision,
                          "recall": recall, "f1": f1}


if __name__ == "__main__":
    df = generate_spam_ham_dataset()
    df.to_csv("spam_ham_data.csv", index=False)
    print(f"Generated synthetic spam/ham dataset: {len(df)} messages "
          f"({(df['label'] == 'spam').sum()} spam, {(df['label'] == 'ham').sum()} ham)")
    print(df.head())

    print("\nPreprocessing text (tokenize, remove stopwords, stem)...")
    df["processed"] = df["message"].apply(preprocess_text)
    print(df[["message", "processed"]].head(3).to_string(index=False))

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["processed"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"\nTF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    results = []
    nb_preds, nb_result = evaluate_model(
        MultinomialNB(), X_train, X_test, y_train, y_test, "Naive Bayes"
    )
    results.append(nb_result)

    print("\nNaive Bayes - Detailed classification report:")
    print(classification_report(y_test, nb_preds))
    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_test, nb_preds, labels=["ham", "spam"])
    print(pd.DataFrame(cm, index=["ham", "spam"], columns=["ham", "spam"]))

    log_preds, log_result = evaluate_model(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test, "Logistic Regression"
    )
    results.append(log_result)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv("nlp_model_comparison.csv", index=False)

    print("\n" + "=" * 50)
    print("Model comparison (best F1 first):")
    print(results_df.to_string(index=False))
