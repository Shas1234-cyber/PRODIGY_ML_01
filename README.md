This project was developed as part of the **Prodigy InfoTech Machine Learning Internship (Task-01)**.

The objective is to build a **Linear Regression Model** that predicts house prices based on:

* Living Area (Square Footage)
* Number of Bedrooms
* Number of Bathrooms

The project uses the **Kaggle House Prices Dataset** and includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization.

---

## Dataset

Dataset: House Prices - Advanced Regression Techniques (Kaggle)

Target Variable:

* SalePrice

Features Used:

* GrLivArea (Above Ground Living Area)
* BedroomAbvGr (Number of Bedrooms)
* FullBath (Number of Full Bathrooms)

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn


---

## Project Structure

```text
HousePricePrediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
├── reports/
│   ├── correlation_heatmap.png
│   ├── distributions.png
│   ├── scatter_vs_target.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── src/
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

### Data Preprocessing

* Missing value handling
* Outlier removal using Z-score
* Feature engineering
* Multicollinearity analysis using VIF

### Exploratory Data Analysis

* Correlation Heatmap
* Feature Distributions
* Scatter Plots
* Statistical Summary

### Model Training

* Linear Regression
* StandardScaler Pipeline
* Train-Test Split
* Cross Validation

### Evaluation Metrics

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## Results

Model Performance on Kaggle Dataset:

| Metric   | Value  |
| -------- | ------ |
| MAE      | 34,571 |
| RMSE     | 46,363 |
| R² Score | 0.53   |

The model explains approximately 53% of the variance in house prices using the selected features.

---

## Generated Reports

The project automatically generates:

* Correlation Heatmap
* Feature Distribution Plots
* Feature vs Target Scatter Plots
* Actual vs Predicted Plot
* Residual Analysis Plot

These reports are saved inside the `reports/` folder.
--

## Internship Information

**Internship:** Prodigy InfoTech Machine Learning Internship

**Task:** Task-01

**Project:** House Price Prediction using Linear Regression

---
