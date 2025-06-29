# Credit Risk Probability Model for Alternative Data

## Project Overview
This repository implements a credit risk model for Bati Bank's "buy-now-pay-later" service, using transaction data from the Xente Fraud Detection Challenge. The model predicts default likelihood, assigns credit scores, and optimizes loan terms, ensuring Basel II compliance. The project follows a modular, reusable structure with clear documentation.

## Setup Instructions
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [Zindi](https://zindi.africa/competitions/xente-fraud-detection-challenge) and place `TrainingData.csv` in `data/`.
4. Run the EDA notebook: `jupyter notebook notebooks/EDA.ipynb`
5. For future scripts, run: `python src/data_processing.py`

## Directory Structure
- `data/`: Raw and processed datasets (excluded from Git; see `.gitignore`).
- `notebooks/`: `EDA.ipynb` for exploratory analysis.
- `src/`: Modular Python scripts for data processing (`data_processing.py`), modeling (`train.py`), and inference (`predict.py`).
- `tests/`: Unit tests (`test_data_processing.py`).
- `api/`: FastAPI application (`main.py`, `pydantic_models.py`).
- `reports/`: Interim (`interim_report.md`) and final reports.
- Root: `README.md`, `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.

## How to Run EDA
1. Ensure dependencies are installed (`requirements.txt`).
2. Open `notebooks/EDA.ipynb` in Jupyter and execute all cells to explore the dataset, calculate RFM metrics, and visualize patterns.

## Interim Submission
- **EDA**: Completed in `notebooks/EDA.ipynb`, including RFM metrics (Recency, Frequency, Monetary), visualizations (histograms, box plots, pair plots), and insights on missing values/outliers.
- **Report**: `reports/interim_report.md` covers credit risk, Basel II, dataset overview, EDA findings, and proxy variable plans.
- **Progress**: Initial feature engineering and proxy variable logic in `src/data_processing.py`, to be finalized for the final submission.