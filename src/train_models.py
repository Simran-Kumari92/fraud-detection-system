import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import joblib

# -------------------------------
# 🚀 Step 1: Start
# -------------------------------
print("🚀 Script started...")

# -------------------------------
# 📂 Step 2: Load dataset
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "../data/AIML Dataset.csv")

print("📂 Loading dataset...")

df = pd.read_csv(data_path, nrows=200000)

print("✅ Dataset loaded:", df.shape)

# -------------------------------
# 🧹 Step 3: Preprocessing
# -------------------------------
df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# 🔥 Feature Engineering
df_model["balance_error"] = (
    df_model["oldbalanceOrg"]
    - df_model["amount"]
    - df_model["newbalanceOrig"]
)

df_model["balance_ratio"] = df_model["amount"] / (df_model["oldbalanceOrg"] + 1)

df_model["is_full_withdrawal"] = (df_model["newbalanceOrig"] == 0).astype(int)



# 🔥 FEATURE ENGINEERING
df_model["balance_error"] = (
    df_model["oldbalanceOrg"]
    - df_model["amount"]
    - df_model["newbalanceOrig"]
)

X = df_model.drop("isFraud", axis=1)
y = df_model["isFraud"]

print("🎯 Target distribution:\n", y.value_counts())

# -------------------------------
# ✂️ Step 4: Train-test split
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("✅ Data split done")

# -------------------------------
# ⚙️ Step 5: Preprocessing
# -------------------------------
categorical = ["type"]

# 🔥 UPDATED numeric features
numeric = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_error",
    "balance_ratio",
    "is_full_withdrawal"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first"), categorical)
])

print("\n🔄 Applying preprocessing...")

x_train_transformed = preprocessor.fit_transform(x_train)
x_test_transformed = preprocessor.transform(x_test)

# -------------------------------
# ⚖️ Step 6: Apply SMOTE
# -------------------------------
print("\n⚖️ Applying SMOTE...")

smote = SMOTE(random_state=42)

x_train_resampled, y_train_resampled = smote.fit_resample(
    x_train_transformed, y_train
)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_resampled.value_counts())

# -------------------------------
# 🤖 Step 7: Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = []

print("\n🚀 Starting model training...")

# -------------------------------
# 🔁 Step 8: Train & Evaluate
# -------------------------------
for name, model in models.items():
    print(f"🔹 Training {name}...")

    model.fit(x_train_resampled, y_train_resampled)
    y_pred = model.predict(x_test_transformed)

    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"]
    })

# -------------------------------
# 📊 Step 9: Save results
# -------------------------------
results_df = pd.DataFrame(results)

print("\n📊 Model Comparison Results:")
print(results_df)

results_path = os.path.join(BASE_DIR, "../results")
os.makedirs(results_path, exist_ok=True)

file_path = os.path.join(results_path, "model_comparison.csv")
results_df.to_csv(file_path, index=False)

print(f"\n✅ Results saved at: {file_path}")

# -------------------------------
# 💾 Step 10: Save best model
# -------------------------------
print("\n💾 Saving best model (Random Forest)...")

best_model = RandomForestClassifier(n_estimators=100, random_state=42)

best_model.fit(x_train_resampled, y_train_resampled)

models_path = os.path.abspath(os.path.join(BASE_DIR, "../models"))
os.makedirs(models_path, exist_ok=True)

model_file = os.path.join(models_path, "fraud_detection_pipeline.pkl")

joblib.dump({
    "model": best_model,
    "preprocessor": preprocessor
}, model_file)

print("✅ Model + Preprocessor saved successfully!")