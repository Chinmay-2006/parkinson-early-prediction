import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

df = pd.read_csv("data/pd_speech_features.csv")
groups = df["id"]
y = df["class"]
X = df.drop(columns=["id", "class"])

# ---------- 1. Random split vs Person-wise split (Random Forest) ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
random_acc = accuracy_score(yte, rf.predict(Xte))

person_acc = cross_val_score(
    RandomForestClassifier(random_state=42), X, y,
    groups=groups, cv=GroupKFold(n_splits=5)
).mean()

print("Random split accuracy:", random_acc)
print("Person-wise accuracy:", person_acc)

# ---------- 2. Bar chart: the "research gap" visual ----------
plt.figure(figsize=(6,4))
plt.bar(["Random Split\n(inflated)", "Person-wise Split\n(honest)"],
        [random_acc*100, person_acc*100], color=["#e74c3c", "#27ae60"])
plt.ylabel("Accuracy (%)")
plt.title("Impact of Data Leakage on Reported Accuracy")
plt.ylim(0, 100)
for i, v in enumerate([random_acc*100, person_acc*100]):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.close()

# ---------- 3. Confusion matrix (Logistic Regression) ----------
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)
lr = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
cm = confusion_matrix(yte, lr.predict(Xte_s))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy","PD"], yticklabels=["Healthy","PD"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------- 4. SHAP explainability ----------
explainer = shap.Explainer(lr, Xtr_s, feature_names=X.columns)
shap_values = explainer(Xte_s[:100])

plt.figure()
shap.summary_plot(shap_values, Xte.iloc[:100], show=False, max_display=10)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()

print("\nDone. Check these files in your folder:")
print("- accuracy_comparison.png")
print("- confusion_matrix.png")
print("- shap_summary.png")