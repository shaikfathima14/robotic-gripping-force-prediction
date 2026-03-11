import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("robotic_grip_dataset_15000.csv")   # <-- your new file name

print("Dataset Loaded Successfully!")
print(data.head())
print("\nShape:", data.shape)

# ----------------------------
# 2. Select Features and Target
# ----------------------------
X = data[['weight', 'friction', 'contact_area', 'jaw_gap',
          'pressure', 'torque', 'hardness', 'stiffness']]

y = data['required_grip_force']

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Linear Regression
# ----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ----------------------------
# 5. Random Forest
# ----------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ----------------------------
# 6. Error Metrics Function
# ----------------------------
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    accuracy = (1 - (mae / y_true.mean())) * 100

    print(f"\n===== {model_name} =====")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Accuracy (%):", accuracy)

# Print both model results
calculate_metrics(y_test, lr_pred, "Linear Regression")
calculate_metrics(y_test, rf_pred, "Random Forest")

# ----------------------------
# 8. Final Predicted Grip Force (for gripper)
# ----------------------------
new_object = [{
    'weight': 65,
    'friction': 0.15,
    'contact_area': 15,
    'jaw_gap': 0,
    'pressure': 110,
    'torque': 3.2,
    'hardness': 5,
    'stiffness': 10
}]

new_object_df = pd.DataFrame(new_object)

final_prediction = rf.predict(new_object_df)[0]

print("\nFinal Predicted Grip Force for New Object:", final_prediction, "Newtons")
