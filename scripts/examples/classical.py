import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_data(jsonl_path: str) -> tuple[list[str], list[int]]:
    """
    Load text data and labels from a prepared JSONL file. The JSONL file is
    expected to have a 'text' field for the raw text and a 'tokens' field for
    the tokenized version. For demonstration purposes, this function generates
    dummy binary labels based on the length of the text (even vs. odd length).

    Parameters
    ----------
    jsonl_path : str
        Path to the prepared JSONL file.

    Returns
    -------
    tuple[list[str], list[int]]
        A tuple containing a list of texts and a list of corresponding labels.
    """

    texts, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            # For demo, use a dummy label: e.g., even/odd length as binary class
            labels.append(len(obj["text"]) % 2)
    return texts, labels


def run_baselines(jsonl_path: str) -> None:
    """
    Run classical NLP baselines (SVM and Logistic Regression) on the prepared
    dataset. The dataset is loaded from a JSONL file, vectorized using TF-IDF,
    and then used to train and evaluate the models.

    Parameters
    ----------
    jsonl_path : str
        Path to the prepared JSONL file containing the dataset.
    """

    texts, labels = load_data(jsonl_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # SVM
    svm = SVC()
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm))

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred_lr))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Run classical NLP baselines on prepared data."
    )
    parser.add_argument(
        "--prepared_jsonl",
        type=str,
        required=True,
        help="Path to prepared JSONL data"
    )
    args = parser.parse_args()
    run_baselines(args.prepared_jsonl)
