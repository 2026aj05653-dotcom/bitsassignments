"""Self-contained Streamlit evaluation app.
Run: python -m streamlit run app.py
Required: models/ with all five trusted trained pickle files beside this file.
Optional: sample_test_data.csv with held-out rows beside this file.
No evaluation.py import and no model training are required.
"""
"""Load trusted saved models and evaluate labelled CSV test data; no training."""
import csv
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)

BASE = Path(__file__).resolve().parent
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'knn.pkl',
    'Gaussian Naive Bayes': 'naive_bayes.pkl',
    'Support Vector Machine': 'svm.pkl',
}
MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000


def load_models():
    model_dir = BASE / 'models'
    missing = [filename for filename in MODEL_FILES.values()
               if not (model_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved model files in {model_dir}: {', '.join(missing)}. "
            "Extract the supplied package and commit the models/ folder beside app.py "
            "in your GitHub repository, then reboot the Streamlit app. "
            "The app cannot generate missing models and never trains on uploaded data."
        )
    models = {}
    for name, filename in MODEL_FILES.items():
        with (BASE / 'models' / filename).open('rb') as file:
            models[name] = pickle.load(file)
    return models


def parse_test_csv(raw, feature_names):
    if len(raw) > MAX_BYTES:
        raise ValueError('CSV must be at most 5 MB. Upload test data only.')
    try:
        text = raw.decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError('Upload a nonempty UTF-8 CSV with a header.') from exc
    if len(set(header)) != len(header):
        raise ValueError('Duplicate column names are not allowed.')
    expected = list(feature_names) + ['malignant']
    missing = sorted(set(expected) - set(header))
    extra = sorted(set(header) - set(expected))
    if missing or extra:
        raise ValueError(f'Column mismatch. Missing: {missing}. Unexpected: {extra}. Use the sample CSV headers.')
    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError('CSV could not be parsed. Use comma-separated columns.') from exc
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError('Upload between 1 and 5,000 test rows.')
    try:
        frame = frame.apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('All features and the target must be numeric.') from exc
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Missing values and infinite values are not allowed.')
    if not frame['malignant'].isin([0, 1]).all():
        raise ValueError('Target malignant must be 0 (benign) or 1 (malignant).')
    return frame.loc[:, list(feature_names)], frame['malignant'].astype(int)


def evaluate_models(models, X, y):
    rows, details = [], {}
    for name, model in models.items():
        predicted = model.predict(X)
        if hasattr(model, 'predict_proba'):
            pos = list(model.classes_).index(1)
            scores = model.predict_proba(X)[:, pos]
        else:
            if list(model.classes_) != [0, 1]:
                raise ValueError('Unexpected saved model class order.')
            scores = model.decision_function(X)
        rows.append({'Model': name,
            'Accuracy': accuracy_score(y, predicted),
            'AUC': roc_auc_score(y, scores) if y.nunique() == 2 else np.nan,
            'Precision': precision_score(y, predicted, pos_label=1, zero_division=0),
            'Recall': recall_score(y, predicted, pos_label=1, zero_division=0),
            'F1': f1_score(y, predicted, pos_label=1, zero_division=0),
            'MCC': matthews_corrcoef(y, predicted)})
        details[name] = {
            'matrix': confusion_matrix(y, predicted, labels=[0, 1]),
            'report': pd.DataFrame(classification_report(y, predicted, labels=[0, 1],
                target_names=['Benign (0)', 'Malignant (1)'], output_dict=True,
                zero_division=0)).T,
            'predictions': pd.DataFrame({'actual_malignant': y.to_numpy(),
                'predicted_malignant': predicted, 'malignant_score': scores}),
        }
    return pd.DataFrame(rows).set_index('Model'), details


"""Launch: python -m streamlit run app.py"""
import numpy as np
import streamlit as st

st.set_page_config(page_title='Classification Model Comparison', layout='wide')

@st.cache_resource
def saved_models():
    return load_models()


def main():
    st.title('Classification Model Comparison')
    st.write('Upload labelled test data to compare five saved classifiers.')
    st.caption('UCI Breast Cancer Wisconsin (Diagnostic): 30 input features. Positive class: malignant = 1; benign = 0.')
    st.markdown('[Dataset source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)')
    st.info('Models were trained beforehand. This app only loads models and predicts; uploaded rows are never used for training.')
    try:
        models = saved_models()
    except Exception as exc:
        st.error(f'Cannot load saved models: {exc}')
        st.caption('If the files exist but loading fails, use the training Python and package versions listed in validated_environment.txt.')
        st.stop()
    selected = st.sidebar.selectbox('Select model for detailed results', list(models))
    features = list(next(iter(models.values())).feature_names_in_)
    sample_path = BASE / 'sample_test_data.csv'
    if sample_path.is_file():
        st.download_button('Download sample held-out test CSV (114 rows)',
            sample_path.read_bytes(), file_name='sample_test_data.csv', mime='text/csv')
    else:
        st.caption('Upload your own held-out test CSV. Add sample_test_data.csv beside app.py to offer a sample download.')
    with st.expander('Required CSV format'):
        st.write('Include exactly these 30 numeric feature columns plus malignant (0 or 1). Column order may vary. No index or patient ID column. Upload held-out test rows only; maximum 5 MB and 5,000 rows.')
        st.code(','.join(features + ['malignant']))
        st.caption('The sample contains the original 114 held-out observations. The app can validate the schema, but cannot determine whether arbitrary uploaded rows were used in training.')
    uploaded = st.file_uploader('Upload test data CSV (features and target required)', type=['csv'])
    if uploaded is None:
        st.info('Upload a labelled test CSV using the required headers shown above.')
        return
    try:
        X, y = parse_test_csv(uploaded.getvalue(), features)
        with st.spinner('Evaluating saved models…'):
            results, details = evaluate_models(models, X, y)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f'Evaluated all five saved models on {len(X)} uploaded test rows.')
    if y.nunique() < 2:
        st.warning('ROC AUC is undefined because the uploaded target has only one class. It is shown as N/A. Include both classes to calculate all six metrics.')
    st.subheader('All-model results on uploaded test data')
    st.dataframe(results.style.format('{:.4f}', na_rep='N/A'))
    st.download_button('Download all evaluation metrics', results.to_csv().encode(),
        file_name='uploaded_test_metrics.csv', mime='text/csv')
    st.subheader(selected)
    for col, (metric, value) in zip(st.columns(6), results.loc[selected].items()):
        col.metric(metric, 'N/A' if np.isnan(value) else f'{value:.4f}')
    st.subheader('Confusion matrix')
    import pandas as pd
    st.dataframe(pd.DataFrame(details[selected]['matrix'],
        index=['Actual benign (0)', 'Actual malignant (1)'],
        columns=['Predicted benign (0)', 'Predicted malignant (1)']))
    st.subheader('Classification report')
    st.dataframe(details[selected]['report'].style.format('{:.4f}'))
    st.download_button('Download selected model predictions',
        details[selected]['predictions'].to_csv(index=False).encode(),
        file_name='test_predictions.csv', mime='text/csv')
    st.caption('AUC uses malignant probabilities or SVM decision scores. SVM decision scores are not probabilities. Precision, recall and F1 use malignant=1; undefined divisions use zero. MCC is zero for a degenerate confusion matrix.')
    with st.expander('Uploaded data preview'):
        st.dataframe(X.assign(malignant=y).head(20))
    st.caption('Academic model comparison; not intended for clinical decisions.')


if __name__ == '__main__':
    main()
