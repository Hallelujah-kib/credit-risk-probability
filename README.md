# Credit Risk Probability Model for Alternative Data

## Project Overview

This repository implements a credit risk model for Bati Bank's "buy-now-pay-later" service, using transaction data from the Xente Fraud Detection Challenge. The goal is to predict customer default likelihood, assign credit scores, and determine optimal loan terms, ensuring compliance with Basel II.

## Setup Instructions

1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Zindi and place in `data/`.
4. Run the EDA notebook: `jupyter notebook notebooks/EDA.ipynb`

## Directory Structure

- `data/`: Raw and processed datasets (excluded from Git).
- `notebooks/`: Jupyter notebooks for EDA (`EDA.ipynb`).
- `src/`: Python scripts for data processing and modeling.
- `tests/`: Unit tests for scripts.
- `api/`: FastAPI application for model deployment.
- `reports/`: Interim and final reports.
- Root: `README.md`, `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.

## How to Run EDA

1. Ensure dependencies are installed.
2. Open `notebooks/EDA.ipynb` in Jupyter and execute all cells to explore the dataset, calculate RFM metrics, and visualize patterns.

## Interim Submission

- EDA completed in `notebooks/EDA.ipynb`, including RFM metrics and visualizations.
- The interim report in `reports/interim_report.md` covers credit risk, Basel II, and initial findings.
- Progress on proxy variable creation is outlined in the report.