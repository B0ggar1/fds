# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
raw_data_path = 'https://raw.githubusercontent.com/B0ggar1/fds/refs/heads/main/movie_details.csv'
preprocessed_data_path = '/mnt/data/preprocessed_movie_details.csv'

raw_data = pd.read_csv(raw_data_path)
preprocessed_data = pd.read_csv(preprocessed_data_path)

# Evaluation scores
evaluation_scores = {
    "Dataset": ["Raw Dataset", "Preprocessed Dataset"],
    "Training Accuracy": [71.36, 85.24],  # Example scores, replace with actual values
    "Cross-Validation Accuracy": [69.89, 83.12]  # Example scores, replace with actual values
}

evaluation_df = pd.DataFrame(evaluation_scores)

# Streamlit app
st.title("Dataset Analysis and Model Evaluation")

# Section 1: Feature Distribution Visualization
st.header("Feature Distributions")

# Feature selection
features = list(raw_data.columns)
selected_feature = st.selectbox("Select a feature to visualize:", features)

# Raw dataset distribution
st.subheader("Raw Dataset Distribution")
fig, ax = plt.subplots()
sns.histplot(raw_data[selected_feature], kde=True, ax=ax, color="blue", label="Raw Data")
plt.legend()
st.pyplot(fig)

# Preprocessed dataset distribution
if selected_feature in preprocessed_data.columns:
    st.subheader("Preprocessed Dataset Distribution")
    fig, ax = plt.subplots()
    sns.histplot(preprocessed_data[selected_feature], kde=True, ax=ax, color="green", label="Preprocessed Data")
    plt.legend()
    st.pyplot(fig)

# Section 2: Comparison of Evaluation Scores
st.header("Evaluation Scores Comparison")
st.subheader("Comparison Matrix")
fig, ax = plt.subplots()
sns.barplot(data=evaluation_df.melt(id_vars="Dataset", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", hue="Dataset", ax=ax)
plt.title("Evaluation Scores: Raw vs Preprocessed")
st.pyplot(fig)

# Footer
st.write("Developed using Streamlit and Seaborn.")
