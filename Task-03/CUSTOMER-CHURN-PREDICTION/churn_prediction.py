import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")
print(classification_report(y_test, lr_pred))
print("Random Forest")
print("Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")
print(classification_report(y_test, rf_pred))
print("Gradient Boosting")
print("Accuracy:", round(accuracy_score(y_test, gb_pred) * 100, 2), "%")
print(classification_report(y_test, gb_pred))
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print("Top Features (Random Forest):")
print(feature_importance)
models = {
    'Logistic Regression': accuracy_score(y_test, lr_pred),
    'Random Forest': accuracy_score(y_test, rf_pred),
    'Gradient Boosting': accuracy_score(y_test, gb_pred)
}
best_model = max(models, key=models.get)
print("Best Model:", best_model, "with accuracy", round(models[best_model] * 100, 2), "%")