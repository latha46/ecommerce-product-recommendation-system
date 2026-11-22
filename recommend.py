
import joblib
import numpy as np
import pandas as pd

# STEP 1: Load trained model
model_data = joblib.load("model/recommendation_model.pkl")

user_factors = model_data["user_factors"]
product_factors = model_data["product_factors"]

user_ids = model_data["user_ids"]
product_ids = model_data["product_ids"]

product_names = model_data["product_names"]
interaction_matrix = model_data["interaction_matrix"]


# STEP 2: Create customer ID lookup
user_lookup = {
    str(uid).replace(".0", ""): index
    for index, uid in enumerate(user_ids)
}


# STEP 3: Recommendation function
def recommend_products(customer_id, top_n=5):

    customer_id = str(customer_id).replace(".0", "").strip()

    # Check whether customer exists
    if customer_id not in user_lookup:
        return "Customer ID not found!"

    # Get customer index
    user_index = user_lookup[customer_id]

    # Predict product preferences
    predicted_scores = (
        user_factors[user_index] @ product_factors.T
    )

    # Identify already purchased products
    purchased = interaction_matrix.iloc[user_index].values > 0

    # Exclude already purchased products
    predicted_scores[purchased] = -np.inf

    # Get top recommended products
    top_indices = np.argsort(predicted_scores)[::-1][:top_n]

    recommendations = []

    for index in top_indices:

        recommendations.append({
            "Product ID": product_ids[index],
            "Product Name": product_names.get(
                product_ids[index], "Unknown Product"
            ),
            "Recommendation Score": round(
                float(predicted_scores[index]), 4
            )
        })

    return pd.DataFrame(recommendations)


# STEP 4: Test recommendation system
customer = str(user_ids[0])

print("\n========== PRODUCT RECOMMENDATIONS ==========")

print("Customer ID:", customer)

results = recommend_products(customer, top_n=5)

print("\nRecommended Products:")
print(results.to_string(index=False))
