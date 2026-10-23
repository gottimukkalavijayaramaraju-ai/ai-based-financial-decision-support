# AI-Based Financial Decision Support

A simple base project with three modules:

1. **Loan Risk** – Gradient Boosting model predicts default probability and explains red flags.
2. **Investment Profile** – questionnaire score gives Conservative / Moderate / Aggressive and an asset allocation.
3. **Budget Check** – compares spending against the 50/30/20 rule.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `model.py` – all logic (data, ML model, rules)
- `app.py` – Streamlit interface

## Next steps
Swap `generate_data()` for a real dataset (e.g. Kaggle "Give Me Some Credit"), add SHAP explanations, stock-price forecasting, or an LLM chat advisor.

*Educational use only; not financial advice.*
