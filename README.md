# E-Commerce Product Recommendation System

## Project Overview

The E-Commerce Product Recommendation System is a machine learning project that recommends relevant products to customers based on their historical purchasing behavior.

It uses Collaborative Filtering and Singular Value Decomposition (SVD) to identify patterns in customer-product interactions and generate personalized Top-5 product recommendations.

The model is integrated with a Streamlit web application that allows users to select a customer and instantly view recommended products with their recommendation scores.

## Project Objectives

- Analyze historical e-commerce customer purchase behavior.
- Build a personalized product recommendation model.
- Apply matrix factorization using SVD.
- Generate Top-5 product recommendations for customers.
- Evaluate recommendation quality using Precision@5, Recall@5 and NDCG@5.
- Deploy the model through an interactive web application.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- Matplotlib / Plotly (as used in the application)

## Dataset

**Dataset:** UCI Online Retail Dataset

The dataset contains historical retail transaction records, including customer IDs, product IDs, quantities and transaction details.

The dataset was cleaned and transformed into customer-product interactions before model training.

## Methodology

### 1. Data Preprocessing
- Loaded the Online Retail dataset.
- Cleaned missing and invalid records.
- Prepared customer-product interaction data.
- Applied log transformation to quantity values using log1p.

### 2. Model Development

Implemented Collaborative Filtering using Singular Value Decomposition (SVD).

The customer-product interaction matrix was factorized into lower-dimensional representations using 50 components.

The trained model identifies purchasing patterns and predicts product preferences.

### 3. Recommendation Generation

- Select a customer ID.
- Retrieve personalized product recommendations.
- Rank products based on predicted recommendation scores.
- Display the Top 5 products.

### 4. Model Evaluation

The model was evaluated using an 80:20 interaction-level train-test split.

Evaluation results:

| Metric | Score |
|---|---:|
| Precision@5 | 0.1575 |
| Recall@5 | 0.1007 |
| NDCG@5 | 0.1921 |

## Dataset and Model Statistics

- Total customer-product interactions: 266,792
- Training interactions: 214,917
- Evaluated customers: 4,247
- SVD components: 50
- Recommendations per customer: 5

## Application Features

- Interactive customer ID selection.
- Personalized product recommendations.
- Top-5 recommended products table.
- Recommendation score visualization.
- Model performance dashboard.

## Project Structure

```text
Ecommerce-Recommendation-System/
│
├── app.py
├── train_model.py
├── recommend.py
├── evaluation.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── cleaned_retail.csv
│
├── model/
│   └── recommendation_model.pkl
│
└── screenshots/
    └── recommendation_dashboard.png
```

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Ecommerce-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

### 4. Open the application

Open the local URL displayed in your terminal.

## Future Improvements

- Explore hybrid recommendation techniques.
- Improve recommendation quality through hyperparameter tuning.
- Incorporate product metadata.
- Add customer segmentation.
- Develop a more scalable recommendation pipeline.

## Conclusion

This project demonstrates the practical application of machine learning in e-commerce personalization. By using collaborative filtering and matrix factorization, the system generates personalized product recommendations and presents them through an interactive Streamlit application.

It demonstrates skills in data preprocessing, machine learning, model evaluation and application development.