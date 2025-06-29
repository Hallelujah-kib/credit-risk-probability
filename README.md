# Credit Risk Probability Model for Alternative Data

## Project Overview
This repository implements a credit risk model for Bati Bank's "buy-now-pay-later" service, using transaction data from the Xente Fraud Detection Challenge. The model predicts default likelihood, assigns credit scores, and optimizes loan terms, ensuring Basel II compliance. The project follows a modular, reusable structure with clear documentation.

## Setup Instructions
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [Zindi](https://zindi.africa/competitions/xente-fraud-detection-challenge) and place `TrainingData.csv` in `data/raw/`.
4. Run the EDA notebook: `jupyter notebook notebooks/1.0-eda.ipynb`
5. For future scripts, run: `python src/data_processing.py`

## Directory Structure
- `data/`: Raw and processed datasets (excluded from Git; see `.gitignore`).
- `notebooks/`: `EDA.ipynb` for exploratory analysis.
- `src/`: Modular Python scripts for data processing (`data_processing.py`), modeling (`train.py`), and inference (`predict.py`).
- `tests/`: Unit tests (`test_data_processing.py`).
- `src/api/`: FastAPI application (`main.py`, `pydantic_models.py`).
- `reports/`: Interim (`interim_report.md`) and final reports.
- Root: `README.md`, `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.

## How to Run EDA
1. Ensure dependencies are installed (`requirements.txt`).
2. Open `notebooks/EDA.ipynb` in Jupyter and execute all cells to explore the dataset, calculate RFM metrics, and visualize patterns.

## Interim Submission
- **EDA**: Completed in `notebooks/EDA.ipynb`, including RFM metrics (Recency, Frequency, Monetary), visualizations (histograms, box plots, pair plots), and insights on missing values/outliers.
- **Report**: `reports/interim_report.md` covers credit risk, Basel II, dataset overview, EDA findings, and proxy variable plans.
- **Progress**: Initial feature engineering and proxy variable logic in `src/data_processing.py`, to be finalized for the final submission.

## 🧩 Regulatory Context and Modeling Considerations

### 🔍 1. Impact of Basel II on Model Interpretability and Documentation

The Basel II Accord places significant emphasis on **quantitative risk measurement** and **regulatory compliance** in credit risk modeling. As a result, financial institutions must ensure that their models are not only accurate but also **interpretable and transparent**. This means:

- Every decision made by the model must be explainable to both internal stakeholders and regulators.
- Model development, assumptions, data preprocessing steps, and validation procedures must be **fully documented**.
- Interpretability becomes critical in **validating model fairness** and ensuring that credit decisions are free from discrimination or bias.

Thus, Basel II encourages the use of **simple and auditable models** and discourages “black-box” models unless their decision paths can be meaningfully explained.

---

### 🛠️ 2. Necessity and Risks of Using a Proxy for Default

Since the dataset lacks a **direct default label** (i.e., whether a customer actually defaulted), we create a **proxy variable**—often based on behavioral indicators like low transaction volume or minimal spending (RFM-based clusters).

While necessary, this introduces **key business risks**:

- **Label Risk**: The proxy may not accurately represent actual defaults, leading to **false positives** or **missed risks**.
- **Model Drift**: The proxy might not generalize well over time or in different segments.
- **Decision Risk**: Using such models to make real credit decisions could result in **biased credit denials**, **lost revenue**, or **regulatory scrutiny**.

To mitigate these risks, any model built on proxies must be treated as **exploratory or indicative**, not as a final decision-making tool until actual outcome labels are collected and validated.

---

### ⚖️ 3. Trade-offs: Interpretable vs. Complex Models in Finance

| Feature | Simple Model (e.g., Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting) |
|--------|--------------------------------------------------|----------------------------------------|
| **Interpretability** | High – easily explainable to regulators and business teams | Low – requires model explainers (e.g., SHAP) |
| **Regulatory Compliance** | Easier to document and audit | Requires extra effort to justify decisions |
| **Performance** | May underperform on complex patterns | Typically higher predictive accuracy |
| **Speed** | Fast to train and deploy | Slower training and tuning |
| **Transparency** | Full control over coefficients and logic | Often seen as a “black box” |

In a regulated financial context, the **best practice** is often to start with **interpretable models** and only layer in complexity if there's a **clear, justifiable performance gain** — and even then, accompanied by **explainability tools** and detailed documentation.
