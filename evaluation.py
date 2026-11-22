
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# STEP 1: Load cleaned dataset
df = pd.read_csv("dataset/cleaned_retail.csv")

df["CustomerID"] = df["CustomerID"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)

print("Dataset loaded successfully!")


# STEP 2: Create customer-product interactions
transactions = df.groupby(
    ["CustomerID", "StockCode"]
)["Quantity"].sum().reset_index()

print("Total customer-product interactions:", len(transactions))


# STEP 3: Split data into training and testing sets
np.random.seed(42)

train_rows = []
test_data = {}

for customer, group in transactions.groupby("CustomerID"):

    products = group["StockCode"].tolist()
    quantities = group["Quantity"].tolist()

    if len(products) < 2:
        continue

    test_count = max(1, int(len(products) * 0.2))
    test_count = min(test_count, len(products) - 1)

    test_indices = np.random.choice(
        len(products),
        size=test_count,
        replace=False
    )

    test_products = set()

    for i in range(len(products)):

        if i in test_indices:
            test_products.add(products[i])

        else:
            train_rows.append({
                "CustomerID": customer,
                "StockCode": products[i],
                "Quantity": quantities[i]
            })

    test_data[customer] = test_products


train_df = pd.DataFrame(train_rows)

print("\nTraining interactions:", len(train_df))
print("Testing customers:", len(test_data))


# STEP 4: Create training interaction matrix
interaction_matrix = train_df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

interaction_matrix = np.log1p(interaction_matrix)

# Keep only customers present in training
test_data = {
    customer: products
    for customer, products in test_data.items()
    if customer in interaction_matrix.index
}

# Keep test products that exist in the training catalogue
test_data = {
    customer: products.intersection(
        set(interaction_matrix.columns)
    )
    for customer, products in test_data.items()
}

test_data = {
    customer: products
    for customer, products in test_data.items()
    if len(products) > 0
}


# STEP 5: Apply Matrix Factorization using SVD
sparse_matrix = csr_matrix(interaction_matrix.values)

n_components = min(
    50,
    min(interaction_matrix.shape) - 1
)

if n_components < 1:
    raise ValueError("Not enough data to train the SVD model.")

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd.fit_transform(sparse_matrix)
product_factors = svd.components_.T

print("\nSVD model trained successfully!")


# STEP 6: Create evaluation metrics
def precision_at_k(recommended, actual, k):

    hits = len(set(recommended[:k]) & actual)

    return hits / k


def recall_at_k(recommended, actual, k):

    hits = len(set(recommended[:k]) & actual)

    return hits / len(actual)


def ndcg_at_k(recommended, actual, k):

    relevance = [
        1 if product in actual else 0
        for product in recommended[:k]
    ]

    dcg = sum(
        rel / np.log2(position + 2)
        for position, rel in enumerate(relevance)
    )

    ideal_hits = min(len(actual), k)

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0


# STEP 7: Generate recommendations and evaluate
precision_scores = []
recall_scores = []
ndcg_scores = []

for user_index, customer in enumerate(
    interaction_matrix.index
):

    if customer not in test_data:
        continue

    actual_products = test_data[customer]

    # Predict customer preferences
    scores = (
        user_factors[user_index] @ product_factors.T
    )

    # Exclude products already purchased in training
    purchased = (
        interaction_matrix.iloc[user_index].values > 0
    )

    scores[purchased] = -np.inf

    # Get top 5 recommendations
    top_indices = np.argsort(scores)[::-1][:5]

    recommended_products = [
        interaction_matrix.columns[i]
        for i in top_indices
    ]

    # Calculate evaluation metrics
    precision_scores.append(
        precision_at_k(
            recommended_products,
            actual_products,
            5
        )
    )

    recall_scores.append(
        recall_at_k(
            recommended_products,
            actual_products,
            5
        )
    )

    ndcg_scores.append(
        ndcg_at_k(
            recommended_products,
            actual_products,
            5
        )
    )


# STEP 8: Display final results
print("\n========== MODEL EVALUATION RESULTS ==========")

print("Evaluated customers:", len(precision_scores))

if len(precision_scores) == 0:
    raise ValueError(
        "No customers had eligible test products. "
        "Check the dataset and train-test split."
    )

print(
    "Precision@5:",
    round(np.mean(precision_scores), 4)
)

print(
    "Recall@5:",
    round(np.mean(recall_scores), 4)
)

print(
    "NDCG@5:",
    round(np.mean(ndcg_scores), 4)
)

print("\nEvaluation completed successfully!")
