from sklearn.ensemble import RandomForestClassifier
import joblib
from .data_processing import load_data, preprocess

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2f}")

joblib.dump(model, 'models/iris_classifier.joblib')
print("Model saved to models/iris_classifier.joblib")
