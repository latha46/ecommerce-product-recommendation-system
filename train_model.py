
import pandas as pd
import numpy as np
import joblib
import os

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# STEP 1: Load cleaned dataset
df = pd.read_csv("dataset/cleaned_retail.csv")

print("Dataset loaded successfully!")

# STEP 2: Create customer-product interaction matrix
interaction_matrix = df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("\nCustomer-Product Matrix Shape:")
print(interaction_matrix.shape)

# STEP 3: Apply log transformation
interaction_matrix = np.log1p(interaction_matrix)

# STEP 4: Convert matrix into sparse format
sparse_matrix = csr_matrix(interaction_matrix.values)

# STEP 5: Apply Matrix Factorization using SVD
n_components = min(50, min(interaction_matrix.shape) - 1)

svd_model = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd_model.fit_transform(sparse_matrix)
product_factors = svd_model.components_.T

# STEP 6: Product descriptions
product_names = (
    df.groupby("StockCode")["Description"]
    .agg(lambda x: x.mode().iloc[0])
    .reindex(interaction_matrix.columns)
    .fillna("Unknown Product")
    .to_dict()
)

# STEP 7: Save trained model
os.makedirs("model", exist_ok=True)

model_data = {
    "svd_model": svd_model,
    "user_factors": user_factors,
    "product_factors": product_factors,
    "user_ids": interaction_matrix.index.tolist(),
    "product_ids": interaction_matrix.columns.tolist(),
    "product_names": product_names,
    "interaction_matrix": interaction_matrix
}

joblib.dump(model_data, "model/recommendation_model.pkl")

# STEP 8: Display results
print("\n========== MODEL TRAINING RESULTS ==========")

print("Total customers:", interaction_matrix.shape[0])
print("Total products:", interaction_matrix.shape[1])
print("SVD components:", n_components)

print(
    "Explained variance ratio:",
    round(svd_model.explained_variance_ratio_.sum(), 4)
)

print("\nModel trained and saved successfully!")
