#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


# Load your dataset
df = pd.read_csv("Downloads/loan_data~.csv")


# In[3]:


print(" First 5 rows of data:\n", df.head())
print("\n🔍 Checking missing values:\n", df.isnull().sum())


# In[4]:


# Fill missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)


# In[5]:


# Encode the only categorical column: 'purpose'
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])


# In[6]:


# Feature and target
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']


# In[7]:


# Save feature order (VERY IMPORTANT)
FEATURE_ORDER = X.columns.tolist()



# In[8]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)


# In[11]:


y_pred = model.predict(X_test_scaled)


# In[12]:


# Evaluate
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


# In[13]:


# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# In[14]:


# 9. FUNCTION: Validate & Predict New Customer
# ======================================================
def predict_new_customer(customer_dict):
    """
    Takes raw customer input, validates it, scales it,
    and predicts loan eligibility.
    """

    # Convert to DataFrame
    customer_df = pd.DataFrame([customer_dict])

    # ---- Column Check ----
    if set(customer_df.columns) != set(FEATURE_ORDER):
        print("Invalid input features")
        print("Expected:", FEATURE_ORDER)
        print("Received:", customer_df.columns.tolist())
        return

    # ---- Reorder Columns ----
    customer_df = customer_df[FEATURE_ORDER]

    # ---- Range Validation (Basic Sanity Check) ----
    for col in FEATURE_ORDER:
        if customer_df[col].iloc[0] < X[col].min() or customer_df[col].iloc[0] > X[col].max():
            print(f" Warning: '{col}' value is outside training range")

    # ---- Scaling ----
    customer_scaled = scaler.transform(customer_df)

    # ---- Prediction ----
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0]

    print("\n Prediction Result")
    print("-------------------")
    if prediction == 0:
        print("Loan likely to be FULLY PAID (Eligible)")
    else:
        print(" Loan likely to DEFAULT (Not Eligible)")

    print(f"Confidence → Fully Paid: {probability[0]*100:.2f}% | Default: {probability[1]*100:.2f}%")

# =


# In[21]:


new_customer = {
    'credit.policy': 1,
    'int.rate': 0.09,
    'installment': 250,
    'log.annual.inc': 11.3,
    'dti': 12,
    'fico': 750,
    'days.with.cr.line': 5500,
    'revol.bal': 8000,
    'revol.util': 30,
    'inq.last.6mths': 0,
    'delinq.2yrs': 0,
    'pub.rec': 0,
    'purpose': le.transform(['credit_card'])[0]
}

predict_new_customer(new_customer)


# In[ ]:




