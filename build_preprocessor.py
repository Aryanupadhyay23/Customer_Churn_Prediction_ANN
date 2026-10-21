import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_and_save_preprocessor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "Dataset", "Churn_Modelling.csv")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load dataset
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Drop irrelevant identifier columns
    X = df.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
    y = df["Exited"]
    
    # Split using the exact same random_state as used during training
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Define the unified ColumnTransformer
    # Output order strictly aligns with the 12 features expected by the trained ANN:
    # 1. CreditScore (scaled)
    # 2. Gender (0: Female, 1: Male)
    # 3-6. Age, Tenure, Balance, NumOfProducts (scaled)
    # 7-8. HasCrCard, IsActiveMember (passthrough binary flags)
    # 9. EstimatedSalary (scaled)
    # 10-12. Geography_France, Geography_Germany, Geography_Spain (OneHot encoded)
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale_credit', StandardScaler(), ['CreditScore']),
            ('gender_enc', OrdinalEncoder(categories=[['Female', 'Male']]), ['Gender']),
            ('scale_demographics', StandardScaler(), ['Age', 'Tenure', 'Balance', 'NumOfProducts']),
            ('binary_flags', 'passthrough', ['HasCrCard', 'IsActiveMember']),
            ('scale_salary', StandardScaler(), ['EstimatedSalary']),
            ('geo_enc', OneHotEncoder(categories=[['France', 'Germany', 'Spain']], sparse_output=False, handle_unknown='ignore'), ['Geography']),
        ]
    )
    
    # 3. Fit transformer
    print("Fitting ColumnTransformer on training features...")
    preprocessor.fit(X_train)
    
    # 4. Save preprocessor artifact
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
        
    print(f"Successfully saved single ColumnTransformer pipeline to {preprocessor_path}")

if __name__ == "__main__":
    build_and_save_preprocessor()
