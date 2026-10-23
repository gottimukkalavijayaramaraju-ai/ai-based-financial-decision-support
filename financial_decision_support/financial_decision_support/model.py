"""Core logic for AI-Based Financial Decision Support.

Three modules:
  1. Credit / loan risk prediction  (machine learning)
  2. Investment risk profiling       (scoring + allocation rules)
  3. Budget analysis                 (50/30/20 rule)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["age", "monthly_income", "credit_score", "monthly_debt",
            "loan_amount", "employment_years", "savings"]


# ---------------------------------------------------------------- 1. CREDIT
def generate_data(n=4000, seed=42):
    """Create synthetic applicant data (replace with a real CSV if you have one)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 65, n),
        "monthly_income": rng.normal(60000, 25000, n).clip(15000, 250000),
        "credit_score": rng.normal(700, 80, n).clip(300, 900),
        "monthly_debt": rng.normal(12000, 8000, n).clip(0, 100000),
        "loan_amount": rng.normal(500000, 300000, n).clip(50000, 3000000),
        "employment_years": rng.integers(0, 35, n),
        "savings": rng.normal(200000, 150000, n).clip(0, 2000000),
    })
    dti = df.monthly_debt / df.monthly_income           # debt-to-income
    lti = df.loan_amount / (df.monthly_income * 12)     # loan-to-annual-income
    z = (-1.0 + 4.0 * dti + 1.2 * lti
         - 0.012 * (df.credit_score - 650)
         - 0.05 * df.employment_years
         - 0.000002 * df.savings
         + rng.normal(0, 0.6, n))
    df["default"] = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    return df


def train_credit_model(df=None):
    df = generate_data() if df is None else df
    X, y = df[FEATURES], df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    metrics = {
        "accuracy": accuracy_score(y_te, model.predict(X_te)),
        "auc": roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]),
    }
    return model, metrics


def assess_loan(model, applicant: dict):
    """Return default probability, decision, and plain-language reasons."""
    p = float(model.predict_proba(pd.DataFrame([applicant])[FEATURES])[0, 1])
    decision = "APPROVE" if p < 0.30 else "REVIEW" if p < 0.55 else "REJECT"

    reasons = []
    dti = applicant["monthly_debt"] / applicant["monthly_income"]
    lti = applicant["loan_amount"] / (applicant["monthly_income"] * 12)
    if applicant["credit_score"] < 650:
        reasons.append("Credit score is below 650.")
    if dti > 0.40:
        reasons.append(f"Debt-to-income ratio is high ({dti:.0%}); lenders prefer under 40%.")
    if lti > 3:
        reasons.append(f"Loan is {lti:.1f}x annual income, which is large.")
    if applicant["savings"] < applicant["monthly_income"] * 3:
        reasons.append("Savings cover less than 3 months of income.")
    if applicant["employment_years"] < 2:
        reasons.append("Short employment history.")
    if not reasons:
        reasons.append("No major red flags found.")
    return {"default_probability": p, "decision": decision, "reasons": reasons}


# ------------------------------------------------------ 2. INVESTMENT PROFILE
QUESTIONS = {
    "horizon": ("Investment horizon", ["< 3 years", "3-7 years", "> 7 years"]),
    "reaction": ("If your portfolio fell 20%, you would...",
                 ["Sell everything", "Hold and wait", "Buy more"]),
    "experience": ("Investing experience", ["None", "Some", "Experienced"]),
    "income_stability": ("Income stability", ["Unstable", "Moderate", "Very stable"]),
}
ALLOCATIONS = {
    "Conservative": {"Debt / Bonds": 70, "Equity": 20, "Gold": 5, "Cash": 5},
    "Moderate":     {"Debt / Bonds": 40, "Equity": 50, "Gold": 5, "Cash": 5},
    "Aggressive":   {"Debt / Bonds": 15, "Equity": 75, "Gold": 5, "Cash": 5},
}


def risk_profile(answers: dict, age: int):
    """answers: {key: option index 0-2}. Returns profile name, score, allocation."""
    score = sum(answers.values())                  # 0..8
    score += 2 if age < 35 else 1 if age < 50 else 0   # younger -> more capacity
    profile = "Conservative" if score <= 4 else "Moderate" if score <= 7 else "Aggressive"
    return {"profile": profile, "score": score, "max_score": 10,
            "allocation": ALLOCATIONS[profile]}


# ------------------------------------------------------------- 3. BUDGET
def analyze_budget(income, needs, wants, savings):
    """Compare spending with the 50/30/20 rule and give advice."""
    total = needs + wants + savings
    pct = {"Needs": needs / income * 100, "Wants": wants / income * 100,
           "Savings": savings / income * 100}
    target = {"Needs": 50, "Wants": 30, "Savings": 20}
    advice = []
    if pct["Needs"] > 50:
        advice.append("Needs exceed 50% of income: look at rent, EMIs, and utilities.")
    if pct["Wants"] > 30:
        advice.append("Discretionary spending exceeds 30%: trim subscriptions and dining out.")
    if pct["Savings"] < 20:
        advice.append("Savings are under 20%: automate a monthly transfer on payday.")
    if total > income:
        advice.append(f"You are overspending by {total - income:,.0f} per month.")
    if not advice:
        advice.append("Your budget follows the 50/30/20 guideline. Keep it up!")
    return {"actual_pct": pct, "target_pct": target, "advice": advice}
