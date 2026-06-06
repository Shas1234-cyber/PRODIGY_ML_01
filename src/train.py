"""
============================================================
House Price Prediction - Linear Regression Pipeline
Prodygy Infotech ML Internship | Task 01
Author  : Shashank
College : NIET, Greater Noida (AKTU)
============================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

FEATURES   = ["GrLivArea", "BedroomAbvGr", "FullBath"]
TARGET     = "SalePrice"
MODEL_PATH = "models/house_price_model.pkl"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ═══════════════════════════════════════════════════════════
# CLASS: DataLoader
# ═══════════════════════════════════════════════════════════
class DataLoader:
    """Loads and performs initial inspection of the dataset."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading dataset from: {self.path}")
        df = pd.read_csv(self.path)
        logger.info(f"Dataset shape: {df.shape}")
        return df


# ═══════════════════════════════════════════════════════════
# CLASS: DataPreprocessor
# ═══════════════════════════════════════════════════════════
class DataPreprocessor:
    """Handles missing values, outlier detection and feature engineering."""

    def __init__(self, features: list, target: str):
        self.features = features
        self.target   = target

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        cols   = self.features + [self.target]
        subset = df[cols].copy()
        before = len(subset)
        missing = subset.isnull().sum()
        logger.info(f"Missing values:\n{missing}")
        for col in self.features:
            if subset[col].isnull().any():
                median_val = subset[col].median()
                subset[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled '{col}' nulls with median={median_val:.2f}")
        subset.dropna(subset=[self.target], inplace=True)
        logger.info(f"Rows: {before} → {len(subset)}")
        return subset

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        before   = len(df)
        z_scores = np.abs(stats.zscore(df[self.features + [self.target]]))
        mask     = (z_scores < 3).all(axis=1)
        df_clean = df[mask].copy()
        logger.info(f"Outlier removal: {before - len(df_clean)} rows removed")
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["RoomBathScore"] = df["BedroomAbvGr"] * df["FullBath"]
        df["LogGrLivArea"]  = np.log1p(df["GrLivArea"])
        logger.info("Feature engineering: added RoomBathScore, LogGrLivArea")
        return df

    def check_vif(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        X        = df[cols].copy()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = cols
        vif_data["VIF"]     = [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])]
        logger.info(f"VIF:\n{vif_data.to_string(index=False)}")
        return vif_data


# ═══════════════════════════════════════════════════════════
# CLASS: EDAAnalyzer
# ═══════════════════════════════════════════════════════════
class EDAAnalyzer:
    """EDA with visualizations."""

    def __init__(self, df, features, target, out_dir):
        self.df       = df
        self.features = features
        self.target   = target
        self.out_dir  = out_dir

    def summary_stats(self):
        logger.info(f"\n{self.df[self.features+[self.target]].describe().to_string()}")

    def correlation_heatmap(self):
        cols = self.features + [self.target]
        corr = self.df[cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, square=True)
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.out_dir, "correlation_heatmap.png")
        plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Saved: {path}")

    def distribution_plots(self):
        cols = self.features + [self.target]
        fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
        for ax, col in zip(axes, cols):
            self.df[col].hist(ax=ax, bins=40, color="#4C72B0", edgecolor="white")
            ax.set_title(col, fontsize=11)
            ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
        plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(self.out_dir, "distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        logger.info(f"Saved: {path}")

    def scatter_vs_target(self):
        fig, axes = plt.subplots(1, len(self.features), figsize=(15, 4))
        for ax, feat in zip(axes, self.features):
            ax.scatter(self.df[feat], self.df[self.target],
                       alpha=0.4, color="#4C72B0", s=10)
            ax.set_xlabel(feat); ax.set_ylabel(self.target)
            ax.set_title(f"{feat} vs {self.target}")
        plt.suptitle("Feature vs SalePrice", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.out_dir, "scatter_vs_target.png")
        plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Saved: {path}")

    def run_all(self):
        self.summary_stats()
        self.correlation_heatmap()
        self.distribution_plots()
        self.scatter_vs_target()


# ═══════════════════════════════════════════════════════════
# CLASS: ModelTrainer
# ═══════════════════════════════════════════════════════════
class ModelTrainer:
    """Train, cross-validate, and evaluate Linear Regression."""

    def __init__(self, features, target, seed=42):
        self.features = features
        self.target   = target
        self.seed     = seed
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = self.y_pred = None

    def split(self, df, test_size=0.2):
        X = df[self.features]; y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed)
        logger.info(f"Train={len(self.X_train)}, Test={len(self.X_test)}")

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ])
        logger.info("Pipeline: StandardScaler → LinearRegression")

    def train(self):
        self.pipeline.fit(self.X_train, self.y_train)
        coef = self.pipeline.named_steps["model"].coef_
        intercept = self.pipeline.named_steps["model"].intercept_
        logger.info("Model trained. Coefficients:")
        for f, c in zip(self.features, coef):
            logger.info(f"  {f}: {c:.4f}")
        logger.info(f"  Intercept: {intercept:.4f}")

    def cross_validate(self, cv=5):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        scores = cross_val_score(self.pipeline, X, y, cv=kfold, scoring="r2")
        logger.info(f"CV R2 ({cv}-fold): {scores.round(4)}")
        logger.info(f"  Mean={scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def evaluate(self) -> dict:
        self.y_pred = self.pipeline.predict(self.X_test)
        mae  = mean_absolute_error(self.y_test, self.y_pred)
        mse  = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(self.y_test, self.y_pred)
        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        logger.info("=== Evaluation Metrics ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:,.2f}")
        return metrics

    def save(self, path):
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved: {path}")


# ═══════════════════════════════════════════════════════════
# CLASS: Visualizer
# ═══════════════════════════════════════════════════════════
class Visualizer:
    """Post-training evaluation plots."""

    def __init__(self, y_test, y_pred, out_dir):
        self.y_test  = y_test
        self.y_pred  = y_pred
        self.out_dir = out_dir
        self.resid   = y_test - y_pred

    def actual_vs_predicted(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(self.y_test, self.y_pred, alpha=0.4, color="#2196F3", s=15)
        lims = [min(self.y_test.min(), self.y_pred.min()),
                max(self.y_test.max(), self.y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual SalePrice"); ax.set_ylabel("Predicted SalePrice")
        ax.set_title("Actual vs Predicted", fontsize=13, fontweight="bold")
        ax.legend(); plt.tight_layout()
        path = os.path.join(self.out_dir, "actual_vs_predicted.png")
        plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Saved: {path}")

    def residual_plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(self.y_pred, self.resid, alpha=0.4, color="#FF5722", s=12)
        axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
        axes[0].set_xlabel("Fitted Values"); axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
        axes[1].hist(self.resid, bins=40, color="#9C27B0", edgecolor="white")
        axes[1].set_xlabel("Residuals"); axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.out_dir, "residual_plot.png")
        plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Saved: {path}")

    def run_all(self):
        self.actual_vs_predicted()
        self.residual_plot()


# ═══════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════
def predict_price(gr_liv_area: float, bedroom_abv_gr: int,
                  full_bath: int, model_path: str = MODEL_PATH) -> float:
    """Predict house price from input features."""
    pipeline = joblib.load(model_path)
    X_new    = pd.DataFrame([[gr_liv_area, bedroom_abv_gr, full_bath]],
                            columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
    price = pipeline.predict(X_new)[0]
    logger.info(f"Prediction: GrLivArea={gr_liv_area}, Bed={bedroom_abv_gr}, "
                f"Bath={full_bath} → ${price:,.2f}")
    return price


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main(csv_path="data/train.csv"):
    logger.info("=" * 60)
    logger.info("  HOUSE PRICE PREDICTION | PRODYGY INFOTECH INTERNSHIP")
    logger.info("=" * 60)

    loader  = DataLoader(csv_path)
    df_raw  = loader.load()

    prep = DataPreprocessor(FEATURES, TARGET)
    df   = prep.handle_missing(df_raw)
    df   = prep.remove_outliers(df)
    df   = prep.engineer_features(df)
    prep.check_vif(df, FEATURES)

    eda  = EDAAnalyzer(df, FEATURES, TARGET, REPORT_DIR)
    eda.run_all()

    trainer = ModelTrainer(FEATURES, TARGET, seed=SEED)
    trainer.split(df, test_size=0.2)
    trainer.build_pipeline()
    trainer.train()
    trainer.cross_validate(cv=5)
    metrics = trainer.evaluate()

    viz = Visualizer(trainer.y_test, trainer.y_pred, REPORT_DIR)
    viz.run_all()

    trainer.save(MODEL_PATH)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
    return metrics


if __name__ == "__main__":
    if not os.path.exists("data/train.csv"):
        logger.warning("train.csv not found – generating synthetic data.")
        np.random.seed(42)
        n = 1500
        gr   = np.random.randint(600, 4000, n).astype(float)
        bed  = np.random.randint(1, 6, n).astype(float)
        bath = np.random.randint(1, 4, n).astype(float)
        price = 50000 + 80*gr + 5000*bed + 15000*bath + np.random.normal(0, 20000, n)
        pd.DataFrame({"GrLivArea": gr, "BedroomAbvGr": bed,
                      "FullBath": bath, "SalePrice": price}).to_csv(
            "data/train.csv", index=False)
    main("data/train.csv")