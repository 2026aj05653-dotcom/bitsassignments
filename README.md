# Classification assignment package

## Dataset and task
UCI Breast Cancer Wisconsin (Diagnostic): 569 instances and 30 numeric input features, exceeding the requirements of 500 instances and 12 features. Binary target: malignant=1, benign=0. Features exclude both patient IDs and the target.

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
The included CSV is exported from scikit-learn's bundled copy, with the target remapped as above.

## Contents
- Classification_Assignment.ipynb: explained notebook with actual local execution outputs; fill in your student details.
- classification_assignment.py: equivalent standalone training/evaluation script.
- breast_cancer_dataset.csv: 569 observations, 30 features and the malignant target.
- classification_results.csv: all six metrics at full precision.
- confusion_matrices.csv: TN, FP, FN and TP for each model.
- requirements.txt: dependencies for running the script and notebook.
- validated_environment.txt: exact package versions used for the included results.

## Run
Extract this ZIP before running. Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python classification_assignment.py
```

For the notebook:

```bash
python -m jupyterlab
```

Open Classification_Assignment.ipynb and select Restart Kernel and Run All Cells. No internet access is needed for the dataset after installing dependencies. The script writes results beside itself; the notebook writes them in its working directory.

## Method
All five models share one stratified 80:20 split (455 training, 114 testing), random_state=42. StandardScaler is fitted only on training data in Logistic Regression, KNN and SVM pipelines. Decision Tree and Gaussian Naive Bayes use unscaled inputs. These are fixed baseline configurations; no test-driven parameter tuning is performed.

Precision, recall and F1 use malignant=1 as the positive class. ROC AUC uses malignant probabilities for four models and the continuous decision function for SVM. It is not computed from predicted labels. MCC uses predicted labels. Scores are rounded only for display.

## Locally validated results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

A single held-out split provides a limited estimate of generalization; the best observed model on this split is not universally best. One extra test error changes accuracy by about 0.88 percentage points. Gaussian NB's conditional independence assumption is imperfect for these correlated features. These results support an academic model comparison, not clinical deployment.

## Execution provenance
Included outputs were computed locally, not in BITS Virtual Lab. No lab screenshot or proof of lab execution is included. If your course still requires that proof, execute the notebook in the required lab and capture one genuine screenshot showing the complete final results table and the lab context. The original uploaded notebook is unchanged.
