# TASK-4-Loan-Default-Prediction

# Loan Default Prediction using Machine Learning

This project predicts whether a loan applicant will **default** on a loan using historical financial data. By leveraging classification algorithms like **LightGBM** and **SVM**, the model helps lending institutions **minimize risk** and **optimize loan approvals**.

---

##  Dataset Used

**Lending Club Loan Dataset** (`loan.csv`)
Each row represents a loan applicant with features like income, credit history, employment status, etc.

---

## Objective

To build a **classification model** that predicts whether a loan applicant is **likely to default**:

* Label **N → 1** (Default)
* Label **Y → 0** (Non-Default)

---

##  Project Workflow

### 1. Data Exploration & Cleaning

* Loaded dataset with Pandas
* Mapped loan status to binary: `Y → 0 (non-default)`, `N → 1 (default)`
* Dropped irrelevant identifiers like `Loan_ID`
* Handled missing values using:

  * Mean imputation for numeric columns
  * Mode imputation for categorical columns

### 2. Preprocessing

* Separated features and target variable
* Used `ColumnTransformer` to:

  * Scale numeric features (`StandardScaler`)
  * One-hot encode categorical features (`OneHotEncoder`)
* Created a **Pipeline** to ensure consistent preprocessing

### 3. Class Imbalance Handling

* Used **SMOTE** (Synthetic Minority Oversampling Technique) to generate synthetic default cases and balance the dataset.

### 4.  Model Training

#### LightGBM Classifier

* Used `GridSearchCV` for hyperparameter tuning:

  * `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`
* Pipeline includes:

  * Preprocessing
  * SMOTE
  * LightGBM classifier
* Achieved:

  * **F1 Score:** `~0.55`
  * **AUC:** `~0.70`

#### SVM Classifier

* Trained using linear and RBF kernels
* Tuned `C` and `kernel` using GridSearchCV
* Pipeline includes:

  * Preprocessing
  * SMOTE
  * SVM classifier with probability output
* Achieved:

  * **F1 Score:** `~0.58`
  * **AUC:** `~0.71`

---

## Evaluation Metrics

| Model    | Precision | Recall | F1-Score | ROC AUC |
| -------- | --------- | ------ | -------- | ------- |
| LightGBM | 0.59      | 0.53   | 0.56     | 0.70    |
| SVM      | 0.75      | 0.47   | 0.58     | 0.71    |

* **Precision:** Useful when minimizing false positives (approve only safe loans)
* **Recall:** Helps in identifying more defaulters (useful in risk alerts)
* **F1 Score:** Balanced measure for both precision and recall

---

## Visualization

* **Confusion Matrix** to understand true/false predictions
* **ROC Curve** for both models to visualize classification performance

---

##  Recommendations for Lenders

1. **Prioritize Precision** for approval decisions to reduce financial loss.
2. Use **Recall** for risk monitoring and early interventions.
3. Implement a **tiered approval system** using probability thresholds:

   * Low risk → auto-approve
   * Medium risk → manual review
   * High risk → reject
4. **Retrain periodically** to adapt to changing borrower behaviors.
5. Monitor **real-world model drift** and performance metrics continuously.

---

## Installation

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn lightgbm
```

---

## File Structure

```
Loan-Default-Prediction/
│
├── loan.csv                 # Raw dataset
├── loan_default_model.ipynb # Main notebook
├── README.md                # Project documentation
└── results/                 # Plots and model output (optional)
```

---

## Outcome

* Successfully built and evaluated a predictive system to identify **high-risk loan applicants**
* Provides actionable insights for **data-driven decision-making** in the lending industry

