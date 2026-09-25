import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Dataset Load karna
data = pd.read_csv('salary_data.csv')
print("--- Dataset Sample ---")
print(data.head())

# 2. Features aur Target split karna
X = data[['experience', 'test_score']]
y = data['salary']

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Train karna
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Performance Check
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 6. Model file save karna
joblib.dump(model, 'salary_model.pkl')
print("\nModel saved as 'salary_model.pkl' successfully!")

# 7. Testing sample
sample_exp = 5
sample_score = 85
pred = model.predict([[sample_exp, sample_score]])
print(f"\nPrediction for {sample_exp} yrs exp & {sample_score} score: Rs. {pred[0]:.2f}")