import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Financial Inclusion Dashboard", layout="wide")

sns.set_style("whitegrid")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Train_v2.csv")

    df['bank_account'] = df['bank_account'].map({'Yes':1, 'No':0})
    df = df.drop(['uniqueid'], axis=1)
    df = df.dropna()

    df_original = df.copy()

    # LABELS FOR VISUALS
    df_original['bank_account_label'] = df_original['bank_account'].map({
        0: 'No Bank Account',
        1: 'Has Bank Account'
    })

    df = pd.get_dummies(df, drop_first=True)

    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    X = df.drop('bank_account', axis=1)
    y = df['bank_account']

    # 🔥 FIX: Handle imbalance
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X, y)

    return model, X.columns, df_original

model, columns, df_original = load_data()

# -------------------------------
# TITLE
# -------------------------------
st.title("💳 Financial Inclusion Intelligence Dashboard")
st.markdown("### Data-Driven Insights & Prediction System")

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

total = len(df_original)
included = df_original['bank_account'].sum()
excluded = total - included

col1.metric("Total People", total)
col2.metric("With Bank Account", included)
col3.metric("Without Bank Account", excluded)

st.markdown("---")

# -------------------------------
# MAIN LAYOUT
# -------------------------------
left, right = st.columns([1,2])

# -------------------------------
# LEFT PANEL (INPUT)
# -------------------------------
with left:
    st.header("🔍 Predict Financial Inclusion")

    age = st.slider("Age", 18, 100, 30)
    location = st.selectbox("Location", ["Urban", "Rural"])
    cellphone = st.selectbox("Mobile Access", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    input_data = {col: 0 for col in columns}
    input_data['age_of_respondent'] = age

    if location == "Urban" and 'location_type_Urban' in columns:
        input_data['location_type_Urban'] = 1

    if cellphone == "Yes" and 'cellphone_access_Yes' in columns:
        input_data['cellphone_access_Yes'] = 1

    if gender == "Male" and 'gender_of_respondent_Male' in columns:
        input_data['gender_of_respondent_Male'] = 1

    input_df = pd.DataFrame([input_data])

    if st.button("🚀 Predict Now"):

        # 🔥 FIX: align columns properly
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("✅ Financially Included")
        else:
            st.error("❌ Not Financially Included")

        st.progress(int(prob * 100))
        st.write(f"Probability: {prob:.2f}")

# -------------------------------
# RIGHT PANEL (VISUALS)
# -------------------------------
with right:
    st.header("📊 Data Insights")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Demographics", "Advanced"])

    # TAB 1
    with tab1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='bank_account_label', data=df_original, palette='Blues', ax=ax1)
        ax1.set_title("Financial Inclusion Distribution", fontweight='bold')
        ax1.set_xlabel("Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    # TAB 2
    with tab2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='gender_of_respondent', hue='bank_account_label', data=df_original, ax=ax2)
        ax2.set_title("Gender vs Financial Inclusion", fontweight='bold')
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.countplot(x='location_type', hue='bank_account_label', data=df_original, ax=ax3)
        ax3.set_title("Location vs Financial Inclusion", fontweight='bold')
        st.pyplot(fig3)

    # TAB 3
    with tab3:
        fig4, ax4 = plt.subplots()
        sns.histplot(df_original['age_of_respondent'], kde=True, color='purple', ax=ax4)
        ax4.set_title("Age Distribution", fontweight='bold')
        st.pyplot(fig4)

        st.write("### Insights")
        st.write("- Urban population has higher inclusion")
        st.write("- Mobile access strongly influences inclusion")
        st.write("- Financial inclusion is still low overall")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("MBA ABA Project | Built using Machine Learning & Streamlit")