import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/pd_speech_features.csv")

groups = df["id"]
y = df["class"]
X = df.drop(columns=["id", "class"])

print("Rows:", X.shape)

# Wrong way: random split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
print("Random split accuracy:", accuracy_score(yte, m.predict(Xte)))

# Right way: person-wise split
s = cross_val_score(RandomForestClassifier(random_state=42), X, y,
                    groups=groups, cv=GroupKFold(n_splits=5))
print("Person-wise accuracy:", s.mean())