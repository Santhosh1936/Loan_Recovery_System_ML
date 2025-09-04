# ===============================
# Smart Loan Recovery System with ML
# ===============================

# Step 1: Import Libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 2: Load Dataset
df = pd.read_csv("/content/loan recovery.csv")
print("Data Loaded Successfully ✅")
print(df.head())

# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

# Loan Amount Distribution vs Monthly Income
fig = px.histogram(df, x='Loan_Amount', nbins=30, marginal="violin", opacity=0.7,
                   title="Loan Amount Distribution & Relationship with Monthly Income",
                   labels={'Loan_Amount': "Loan Amount ($)", 'Monthly_Income': "Monthly Income"},
                   color_discrete_sequence=["royalblue"])

fig.add_trace(go.Scatter(
    x=sorted(df['Loan_Amount']),
    y=px.histogram(df, x='Loan_Amount', nbins=30, histnorm='probability density').data[0]['y'],
    mode='lines',
    name='Density Curve',
    line=dict(color='red', width=2)
))

scatter = px.scatter(df, x='Loan_Amount', y='Monthly_Income',
                     color='Loan_Amount', color_continuous_scale='Viridis',
                     size=df['Loan_Amount'], hover_name=df.index)

for trace in scatter.data:
    fig.add_trace(trace)

fig.update_layout(
    annotations=[
        dict(
            x=max(df['Loan_Amount']) * 0.8, y=max(df['Monthly_Income']),
            text="Higher Loan Amounts linked to Higher Income",
            showarrow=True, arrowhead=2, font=dict(size=12, color="red")
        )
    ],
    xaxis_title="Loan Amount ($)", yaxis_title="Monthly Income ($)",
    template="plotly_white", showlegend=True
)
fig.show()

# ===============================
# Payment History Analysis
# ===============================

fig = px.histogram(df, x="Payment_History", color="Recovery_Status", barmode="group",
                   title="How Payment History Affects Loan Recovery Status",
                   labels={"Payment_History": "Payment History", "count": "Number of Loans"})
fig.update_layout(template="plotly_white")
fig.show()

# Missed Payments vs Recovery
fig = px.box(df, x="Recovery_Status", y="Num_Missed_Payments",
             title="How Missed Payments Affect Loan Recovery",
             labels={"Recovery_Status": "Recovery Status", "Num_Missed_Payments": "Missed Payments"},
             color="Recovery_Status", points="all")
fig.update_layout(template="plotly_white")
fig.show()

# ===============================
# Loan Recovery vs Income
# ===============================

fig = px.scatter(df, x='Monthly_Income', y='Loan_Amount',
                 color='Recovery_Status', size='Loan_Amount',
                 hover_data={'Monthly_Income': True, 'Loan_Amount': True, 'Recovery_Status': True},
                 title="Monthly Income vs Loan Amount Recovery")
fig.add_annotation(
    x=max(df['Monthly_Income']), y=max(df['Loan_Amount']),
    text="Higher income helps recover larger loans",
    showarrow=True, arrowhead=2, font=dict(size=12, color="red")
)
fig.update_layout(template="plotly_white")
fig.show()

# ===============================
# Borrower Segmentation (K-Means)
# ===============================

features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
            'Num_Missed_Payments', 'Days_Past_Due']

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Borrower_Segment'] = kmeans.fit_predict(df_scaled)

fig = px.scatter(df, x='Monthly_Income', y='Loan_Amount',
                 color=df['Borrower_Segment'].astype(str), size='Loan_Amount',
                 title="Borrower Segments (Income vs Loan)",
                 labels={"Borrower_Segment": "Segment"},
                 color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_layout(template="plotly_white")
fig.show()

# Rename Segments
df['Segment_Name'] = df['Borrower_Segment'].map({
    0: 'Moderate Income, High Loan Burden',
    1: 'High Income, Low Default Risk',
    2: 'Moderate Income, Medium Risk',
    3: 'High Loan, Higher Default Risk'
})

# ===============================
# Early Default Detection (RF Model)
# ===============================

df['High_Risk_Flag'] = df['Segment_Name'].apply(
    lambda x: 1 if x in ['High Loan, Higher Default Risk', 'Moderate Income, High Loan Burden'] else 0
)

X = df[features]
y = df['High_Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

risk_scores = rf_model.predict_proba(X_test)[:, 1]

df_test = X_test.copy()
df_test['Risk_Score'] = risk_scores
df_test['Predicted_High_Risk'] = (df_test['Risk_Score'] > 0.5).astype(int)

df_test = df_test.merge(df[['Borrower_ID', 'Segment_Name', 'Recovery_Status',
                            'Collection_Method', 'Collection_Attempts', 'Legal_Action_Taken']],
                        left_index=True, right_index=True)

# ===============================
# Dynamic Recovery Strategy
# ===============================

def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"

df_test['Recovery_Strategy'] = df_test['Risk_Score'].apply(assign_recovery_strategy)

print("\n=== Final Predictions with Strategies ===")
print(df_test[['Borrower_ID', 'Risk_Score', 'Predicted_High_Risk',
               'Segment_Name', 'Recovery_Status', 'Recovery_Strategy']].head())
