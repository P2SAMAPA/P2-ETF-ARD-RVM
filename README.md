# Sparse Bayesian ARD‑RVM Engine

Implements the Relevance Vector Machine (RVM) with Automatic Relevance Determination (ARD). The model is a Bayesian linear regression where each feature has its own precision hyperparameter. Type‑II maximum likelihood (evidence maximisation) drives irrelevant hyperparameters to infinity, pruning features (exact zeros). The result is a sparse, fully probabilistic model.

- **Features:** lagged returns (1,2,3,5,10,21 days) + macro levels
- **Algorithm:** Fast EM for RVM (Tipping & Faul, 2003)
- **Sparsity:** automatic feature selection via ARD
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
