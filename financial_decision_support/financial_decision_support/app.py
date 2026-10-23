"""Streamlit UI.  Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import model as fm

st.set_page_config(page_title="AI Financial Decision Support", page_icon="💹")
st.title("💹 AI-Based Financial Decision Support")
st.caption("Educational demo using synthetic data. Not financial advice.")


@st.cache_resource
def load_model():
    return fm.train_credit_model()


clf, metrics = load_model()
tab1, tab2, tab3 = st.tabs(["Loan Risk", "Investment Profile", "Budget Check"])

# ---- Tab 1
with tab1:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 21, 70, 30, key="loan_age")
    income = c1.number_input("Monthly income", 10000, 500000, 60000, 5000)
    score = c1.slider("Credit score", 300, 900, 700)
    debt = c1.number_input("Existing monthly debt (EMIs)", 0, 200000, 10000, 1000)
    loan = c2.number_input("Loan amount requested", 50000, 5000000, 500000, 50000)
    emp = c2.number_input("Years employed", 0, 40, 5)
    sav = c2.number_input("Savings", 0, 5000000, 200000, 10000)

    if st.button("Assess loan"):
        r = fm.assess_loan(clf, dict(age=age, monthly_income=income, credit_score=score,
                                     monthly_debt=debt, loan_amount=loan,
                                     employment_years=emp, savings=sav))
        st.metric("Default probability", f"{r['default_probability']:.1%}")
        {"APPROVE": st.success, "REVIEW": st.warning, "REJECT": st.error}[r["decision"]](
            f"Decision: {r['decision']}")
        for reason in r["reasons"]:
            st.write("•", reason)
    st.caption(f"Model test accuracy {metrics['accuracy']:.1%}, AUC {metrics['auc']:.2f}")

# ---- Tab 2
with tab2:
    age2 = st.number_input("Your age", 18, 80, 30, key="inv_age")
    answers = {}
    for key, (label, options) in fm.QUESTIONS.items():
        answers[key] = options.index(st.radio(label, options, horizontal=True, key=key))
    if st.button("Get my profile"):
        p = fm.risk_profile(answers, age2)
        st.subheader(f"{p['profile']} investor ({p['score']}/{p['max_score']})")
        st.bar_chart(pd.Series(p["allocation"], name="Allocation %"))

# ---- Tab 3
with tab3:
    inc = st.number_input("Monthly income", 1000, 1000000, 60000, 1000, key="b_inc")
    needs = st.number_input("Needs (rent, food, bills, EMIs)", 0, 1000000, 30000, 1000)
    wants = st.number_input("Wants (entertainment, shopping)", 0, 1000000, 15000, 1000)
    sv = st.number_input("Savings / investments", 0, 1000000, 8000, 1000)
    if st.button("Analyze budget"):
        b = fm.analyze_budget(inc, needs, wants, sv)
        st.dataframe(pd.DataFrame({"Actual %": b["actual_pct"],
                                   "Target %": b["target_pct"]}).round(1))
        for a in b["advice"]:
            st.write("•", a)
