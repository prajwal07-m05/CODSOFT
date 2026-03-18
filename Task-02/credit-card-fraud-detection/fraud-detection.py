import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

train = train.sample(frac=0.05, random_state=42)
test = test.sample(frac=0.05, random_state=42)

drop_cols = ['Unnamed: 0','trans_date_trans_time','cc_num','first','last',
             'street','city','state','dob','trans_num']
train = train.drop(columns=drop_cols, errors='ignore')
test = test.drop(columns=drop_cols, errors='ignore')

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train, test = train.align(test, join='left', axis=1, fill_value=0)

X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]
X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

model = RandomForestClassifier(n_estimators=5, max_depth=5, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)