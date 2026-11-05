"""Run with: python -m streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve,
)

st.set_page_config(page_title="Classification Model Comparison", layout="wide")

@st.cache_resource
def train_and_evaluate():
    """Fit the same baselines and split as the assignment, in this environment."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series((data.target == 0).astype(int), name="malignant")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000, random_state=42)),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=42),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=5, weights="uniform", metric="minkowski", p=2)),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", gamma="scale")),
    }
    rows, details = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            index = int(np.flatnonzero(model.classes_ == 1)[0])
            score = model.predict_proba(X_test)[:, index]
        else:
            assert list(model.classes_) == [0, 1]
            score = model.decision_function(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        details[name] = {
            "confusion": confusion_matrix(y_test, pred, labels=[0, 1]),
            "roc": roc_curve(y_test, score),
            "predictions": pd.DataFrame({
                "sample_index": y_test.index, "actual_malignant": y_test.to_numpy(),
                "predicted_malignant": pred, "malignant_score": score,
            }),
        }
    return X, y, pd.DataFrame(rows).set_index("Model"), details


def main():
    st.title("Classification Model Comparison")
    st.write("Five classifiers evaluated on one shared held-out test set.")
    st.caption("UCI Breast Cancer Wisconsin (Diagnostic) · 569 instances · 30 features")
    st.markdown("[Dataset source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")
    with st.spinner("Training and evaluating the five models…"):
        X, y, results, details = train_and_evaluate()
    selected = st.sidebar.selectbox("Select classifier", list(results.index))
    st.sidebar.markdown("**Evaluation settings**")
    st.sidebar.write("Stratified 80:20 split · seed 42\n\n455 training / 114 test observations")
    st.sidebar.caption("Positive class: malignant (1). Benign: 0. Scaling is fitted on training data only.")
    st.sidebar.caption(f"scikit-learn {sklearn.__version__} · Streamlit {st.__version__}")
    st.subheader(selected)
    for col, (metric, value) in zip(st.columns(6), results.loc[selected].items()):
        col.metric(metric, f"{value:.4f}")
    st.subheader("All-model comparison")
    st.dataframe(results.style.format("{:.4f}"))
    st.download_button("Download all metrics (CSV)", results.to_csv().encode(),
                       file_name="classification_results.csv", mime="text/csv")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = pd.DataFrame(details[selected]["confusion"],
            index=["Actual benign", "Actual malignant"],
            columns=["Predicted benign", "Predicted malignant"])
        st.dataframe(matrix)
        st.caption("False negatives are malignant cases predicted benign.")
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = details[selected]["roc"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(fpr, tpr, label=f"AUC = {results.loc[selected, 'AUC']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("AUC uses malignant probabilities, or the SVM decision score. SVM scores are not probabilities.")
    st.download_button("Download selected model's test predictions", details[selected]["predictions"].to_csv(index=False).encode(),
                       file_name="test_predictions.csv", mime="text/csv")
    with st.expander("Dataset preview and class distribution"):
        st.write("Benign:", int((y == 0).sum()), "· Malignant:", int((y == 1).sum()))
        st.dataframe(X.assign(malignant=y).head(20))
        st.bar_chart(y.map({0: "Benign", 1: "Malignant"}).value_counts())
    st.info("Educational model comparison. Scores are from one test split and do not establish clinical suitability.")


if __name__ == "__main__":
    main()
