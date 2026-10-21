import os
import sys
import pickle
import pandas as pd
from tensorflow.keras.models import load_model


class CustomData:
    """
    Encapsulates all incoming customer data from the user interface
    and converts it into a pandas DataFrame ready for the prediction pipeline.
    """
    def __init__(
        self,
        credit_score: int,
        geography: str,
        gender: str,
        age: int,
        tenure: int,
        balance: float,
        num_products: int,
        has_cr_card: int,
        is_active: int,
        salary: float
    ):
        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_products = num_products
        self.has_cr_card = has_cr_card
        self.is_active = is_active
        self.salary = salary

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Returns a single-row DataFrame matching the raw input columns.
        """
        data_dict = {
            "CreditScore": [self.credit_score],
            "Geography": [self.geography],
            "Gender": [self.gender],
            "Age": [self.age],
            "Tenure": [self.tenure],
            "Balance": [self.balance],
            "NumOfProducts": [self.num_products],
            "HasCrCard": [self.has_cr_card],
            "IsActiveMember": [self.is_active],
            "EstimatedSalary": [self.salary]
        }
        return pd.DataFrame(data_dict)


class PredictPipeline:
    """
    Dedicated prediction pipeline that encapsulates model loading,
    feature transformation via ColumnTransformer, and churn probability inference.
    """
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        # Resolve project root regardless of execution directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        self.model_path = model_path or os.path.join(base_dir, "model", "model.h5")
        self.preprocessor_path = preprocessor_path or os.path.join(base_dir, "artifacts", "preprocessor.pkl")

        self.model = None
        self.preprocessor = None

    def _load_artifacts(self):
        """Lazy load model and preprocessor if not already loaded."""
        if self.preprocessor is None:
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at: {self.preprocessor_path}")
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            self.model = load_model(self.model_path)

    def predict(self, features: pd.DataFrame) -> tuple[float, bool]:
        """
        Transforms raw feature DataFrame and executes ANN prediction.

        Returns:
            tuple[float, bool]: (churn_probability, is_churn_predicted)
        """
        self._load_artifacts()

        # Single-line transformation via unified ColumnTransformer
        transformed_features = self.preprocessor.transform(features)

        # ANN prediction
        prob = float(self.model.predict(transformed_features, verbose=0)[0][0])
        is_churn = prob > 0.5

        return prob, is_churn
