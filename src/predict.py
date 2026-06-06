import joblib
import pandas as pd

MODEL_PATH = "models/house_price_model.pkl"

model = joblib.load(MODEL_PATH)

test_cases = [
    {"GrLivArea": 1200, "BedroomAbvGr": 2, "FullBath": 1},
    {"GrLivArea": 1800, "BedroomAbvGr": 3, "FullBath": 2},
    {"GrLivArea": 2500, "BedroomAbvGr": 4, "FullBath": 3},
    {"GrLivArea": 3200, "BedroomAbvGr": 5, "FullBath": 3},
]

print("\nHOUSE PRICE PREDICTIONS\n")

for case in test_cases:
    X = pd.DataFrame([case])

    prediction = model.predict(X)[0]

    print(
        f"Area={case['GrLivArea']} | "
        f"Bedrooms={case['BedroomAbvGr']} | "
        f"Baths={case['FullBath']} "
        f"--> Predicted Price = ${prediction:,.2f}"
    )