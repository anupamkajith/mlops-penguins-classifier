import pandas as pd

from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def load_data():
    """Load and prepare the Palmer Penguins dataset."""
    df = load_penguins()

    # Remove rows containing missing values.
    df = df.dropna().copy()

    X = df.drop(columns=["species"])
    y = df["species"]

    return X, y


def build_pipeline(X):
    """Create preprocessing and Decision Tree classification pipeline."""

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained classifier."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        labels=model.classes_,
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print(f"ROC-AUC (OvR):   {auc:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predictions))


def main():
    X, y = load_data()

    print(f"Dataset size after removing missing values: {len(X)}")
    print(f"Target classes: {sorted(y.unique().tolist())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(X)

    pipeline.fit(X_train, y_train)

    evaluate_model(
        pipeline,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
