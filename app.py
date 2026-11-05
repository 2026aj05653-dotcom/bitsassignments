"""Launch: python -m streamlit run app.py"""
import numpy as np
import streamlit as st
from evaluation import BASE, load_models, parse_test_csv, evaluate_models

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
        st.error(f'Cannot load saved models: {exc}. Include models/ and install the pinned requirements.')
        st.stop()
    selected = st.sidebar.selectbox('Select model for detailed results', list(models))
    features = list(next(iter(models.values())).feature_names_in_)
    st.download_button('Download sample held-out test CSV (114 rows)',
        (BASE / 'sample_test_data.csv').read_bytes(), file_name='sample_test_data.csv', mime='text/csv')
    with st.expander('Required CSV format'):
        st.write('Include exactly these 30 numeric feature columns plus malignant (0 or 1). Column order may vary. No index or patient ID column. Upload held-out test rows only; maximum 5 MB and 5,000 rows.')
        st.code(','.join(features + ['malignant']))
        st.caption('The sample contains the original 114 held-out observations. The app can validate the schema, but cannot determine whether arbitrary uploaded rows were used in training.')
    uploaded = st.file_uploader('Upload test data CSV (features and target required)', type=['csv'])
    if uploaded is None:
        st.info('Download the sample above and upload it here, or upload your own labelled test CSV with the same headers.')
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
