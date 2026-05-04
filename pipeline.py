"""
Loan Default Risk Modeling - Data Preprocessing Pipeline
Description: An object-oriented, strictly encapsulated data preprocessing pipeline. This module mathematically prepares tabular data for parametric machine learning models, enforcing strict isolation between training and holdout datasets to prevent data leakage.
"""

import pandas as pd
from pathlib import Path

# [LIBRARY IMPORTS AND JUSTIFICATIONS]
# sklearn.model_selection.train_test_split is imported to perform randomized data partitioning. It is preferred over custom slicing as it mathematically guarantees proper stochastic shuffling and supports stratified sampling to maintain the prior probability of the target distribution.
from sklearn.model_selection import train_test_split

# sklearn.preprocessing.StandardScaler is utilized to standardize continuous features by removing the mean and scaling to unit variance (z = (x - u) / s). This is mathematically imperative for gradient-based optimization algorithms (e.g., Logistic Regression, Neural Networks) to ensure symmetric and rapid convergence, preventing features with larger magnitudes from dominating the objective function.
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        """
        Initializes the preprocessor with a fixed seed for stochastic reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Defining column taxonomies for robust programmatic referencing
        self.target_col = 'default'
        self.id_col = 'loan_id'
        self.num_cols = ['income', 'loan_amount']
        self.cat_cols = ['employment_status']

    def _drop_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes non-predictive surrogate keys.
        """
        if self.id_col in df.columns:
            # Dropping surrogate keys prevents the model from mapping spurious correlations or memorizing specific instances (overfitting), ensuring the model generalizes based on causal or statistically significant latent features.
            df = df.drop(columns=[self.id_col])
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms categorical taxonomies into numerical, orthogonal vector representations.
        """
        # pandas.get_dummies is deployed for One-Hot Encoding.
        # drop_first=True is rigorously applied to avoid multicollinearity (the 'dummy variable trap'), ensuring the resulting design matrix is strictly full rank, which is required for stable matrix inversion in linear models (e.g., OLS or unregularized Logistic Regression).
        df_encoded = pd.get_dummies(df, columns=self.cat_cols, drop_first=True, dtype=int)
        return df_encoded

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """
        Partitions the dataset into Train (70%), Validation (15%), and Test (15%) sets.
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Splitting strategy: First split isolates 70% for training and 30% for holdout.
        # The 'stratify=y' argument is crucial here to ensure the marginal distribution of the target variable (default rate) is preserved identically across all data splits.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.random_state, stratify=y
        )

        # Second split evenly divides the 30% holdout into 15% Validation and 15% Test.
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Applies z-score normalization strictly utilizing training dataset parameters.
        """
        # DATA LEAKAGE PREVENTION CORE:
        # The StandardScaler is EXCLUSIVELY fitted (.fit()) on X_train. 
        # Calculating the mean and standard deviation on the entire dataset prior to splitting would inject future information (data leakage) into the training process, invalidating the i.i.d. assumption and artificially inflating evaluation metrics.
        
        # We copy the dataframes to prevent SettingWithCopyWarning and preserve original memory states.
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        # .fit_transform calculates parameters (u, s) on Train and applies them simultaneously.
        X_train_scaled[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols])
        
        # .transform strictly applies the PREVIOUSLY computed parameters to the holdout sets.
        X_val_scaled[self.num_cols] = self.scaler.transform(X_val[self.num_cols])
        X_test_scaled[self.num_cols] = self.scaler.transform(X_test[self.num_cols])

        return X_train_scaled, X_val_scaled, X_test_scaled

    def process_data(self, filepath: str) -> tuple:
        """
        Orchestrates the preprocessing pipeline sequentially.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test (DataFrames/Series)
        """
        print("Starting data preprocessing pipeline...")
        
        # 1. Load Data
        df = pd.read_csv(filepath)
        print(f"Initial shape: {df.shape}")
        
        # 2. Drop identifiers
        df = self._drop_identifiers(df)
        
        # 3. Encode categorical variables
        df = self._encode_categorical(df)
        
        # 4. Rigorous splitting (70/15/15)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(df)
        print(f"Data split - Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # 5. Safe feature scaling (Leakage prevention)
        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)
        print("Feature scaling applied successfully without data leakage.")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        output_dir: str = ".",
    ) -> None:
        """
        Exports processed train, validation, and test datasets as CSV files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(output_path / "X_train_processed.csv", index=False)
        X_val.to_csv(output_path / "X_val_processed.csv", index=False)
        X_test.to_csv(output_path / "X_test_processed.csv", index=False)

        y_train.to_csv(output_path / "y_train_processed.csv", index=False)
        y_val.to_csv(output_path / "y_val_processed.csv", index=False)
        y_test.to_csv(output_path / "y_test_processed.csv", index=False)

        print(f"Processed CSV files saved to: {output_path.resolve()}")

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # Initialize the robust preprocessor
    preprocessor = DataPreprocessor(random_state=42)
    
    # Execute the master function
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_data('loan_default_prediction.csv')
    preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Verification output
    print("\nSample of Scaled and Encoded X_train:")
    print(X_train.head(3))
