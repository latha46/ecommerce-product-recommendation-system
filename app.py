
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# PAGE CONFIGURATION
st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    return joblib.load("model/recommendation_model.pkl")


model_data = load_model()

user_factors = model_data["user_factors"]
product_factors = model_data["product_factors"]

user_ids = model_data["user_ids"]
product_ids = model_data["product_ids"]

product_names = model_data["product_names"]
interaction_matrix = model_data["interaction_matrix"]


# CUSTOMER LOOKUP
user_lookup = {
    str(uid).replace(".0", ""): index
    for index, uid in enumerate(user_ids)
}


# RECOMMENDATION FUNCTION
def recommend_products(customer_id, top_n=5):

    customer_id = str(customer_id).strip()

    if customer_id not in user_lookup:
        return None

    user_index = user_lookup[customer_id]

    scores = (
        user_factors[user_index] @ product_factors.T
    )

    purchased = (
        interaction_matrix.iloc[user_index].values > 0
    )

    scores[purchased] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []

    for index in top_indices:
        results.append({
            "Product ID": product_ids[index],
            "Product Name": product_names.get(
                product_ids[index], "Unknown Product"
            ),
            "Recommendation Score": round(
                float(scores[index]), 4
            )
        })

    return pd.DataFrame(results)


# APP HEADER
st.title("🛍️ E-Commerce Product Recommendation System")

st.write(
    "A personalized product recommendation system "
    "powered by Collaborative Filtering and Matrix Factorization."
)

st.divider()


# DASHBOARD METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(user_ids):,}")
col2.metric("Total Products", f"{len(product_ids):,}")
col3.metric("Algorithm", "SVD")

st.divider()


# CUSTOMER SELECTION
st.subheader("Find Personalized Recommendations")

customer_id = st.selectbox(
    "Select Customer ID",
    options=user_ids,
    index=0
)


if st.button("Generate Recommendations", type="primary"):

    recommendations = recommend_products(customer_id)

    if recommendations is not None:

        st.success(
            f"Recommendations generated for Customer {customer_id}"
        )

        # TOP 5 RECOMMENDATIONS
        st.subheader("Top 5 Recommended Products")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        # PRODUCT RECOMMENDATION CHART
        st.subheader("Product Recommendation Scores")

        chart_data = recommendations.sort_values(
            "Recommendation Score",
            ascending=True
        ).copy()

        fig = px.bar(
            chart_data,
            x="Recommendation Score",
            y="Product Name",
            orientation="h",
            text="Recommendation Score",
            hover_data=["Product ID"]
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            height=450,
            margin=dict(l=30, r=80, t=30, b=30),
            yaxis=dict(
                title="Product Name",
                automargin=True,
                tickfont=dict(size=11)
            ),
            xaxis=dict(
                title="Recommendation Score",
                showgrid=True
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.error("Customer ID not found.")


st.divider()


# MODEL EVALUATION
st.subheader("Model Performance")

eval_col1, eval_col2, eval_col3 = st.columns(3)

eval_col1.metric("Precision@5", "0.1575")
eval_col2.metric("Recall@5", "0.1007")
eval_col3.metric("NDCG@5", "0.1921")

st.caption(
    "Evaluation results are based on the 80:20 "
    "interaction-level train-test split."
)

st.divider()


# PROJECT FOOTER
st.markdown(
    "**Project:** E-Commerce Product Recommendation System  \n"
    "**Algorithm:** Collaborative Filtering using Truncated SVD  \n"
    "**Built with:** Python, Pandas, Scikit-learn and Streamlit"
)
