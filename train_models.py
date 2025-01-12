import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


X, y_systolic = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
_, y_diastolic = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=7)


X_train, _, y_systolic_train, _ = train_test_split(X, y_systolic, test_size=0.2, random_state=42)
_, _, y_diastolic_train, _ = train_test_split(X, y_diastolic, test_size=0.2, random_state=42)


systolic_model = LinearRegression().fit(X_train, y_systolic_train)
diastolic_model = LinearRegression().fit(X_train, y_diastolic_train)


joblib.dump(systolic_model, 'bp_model_systolic.pkl')
joblib.dump(diastolic_model, 'bp_model_diastolic.pkl')

print("Models saved successfully!")
