

import pandas as pd
import os

# Define file paths
data_folder = "dataset"
euclidean_path = os.path.join(data_folder, "euclidean_similarity_results.csv")
cosine_path = os.path.join(data_folder, "cosine_similarity_results.csv")
output_path = os.path.join(data_folder, "merged_similarity_results.csv")

# Check if files exist
if not os.path.exists(euclidean_path) or not os.path.exists(cosine_path):
    print("One or both CSV files are missing.")
    exit()

# Load CSV files
euclidean_df = pd.read_csv(euclidean_path)
cosine_df = pd.read_csv(cosine_path)

# Strip spaces from column names
euclidean_df.columns = euclidean_df.columns.str.strip()
cosine_df.columns = cosine_df.columns.str.strip()

# Ensure required columns exist
required_columns = {"text_id", "similarity_score"}
if not required_columns.issubset(euclidean_df.columns) or not required_columns.issubset(cosine_df.columns):
    print("One of the files is missing required columns.")
    exit()

# Rename similarity_score columns
euclidean_df.rename(columns={"similarity_score": "euclidean_similarity_score"}, inplace=True)
cosine_df.rename(columns={"similarity_score": "cosine_similarity_score"}, inplace=True)

# Merge on text_id
merged_df = pd.merge(euclidean_df, cosine_df, on="text_id", how="outer")

# Convert similarity scores to numeric (in case of errors in CSV)
merged_df["euclidean_similarity_score"] = pd.to_numeric(merged_df["euclidean_similarity_score"], errors="coerce")
merged_df["cosine_similarity_score"] = pd.to_numeric(merged_df["cosine_similarity_score"], errors="coerce")

# Create 'is_high_similarity' column: True if both scores > 0.7, else False
merged_df["is_high_similarity"] = (merged_df["euclidean_similarity_score"] > 0.7) & (merged_df["cosine_similarity_score"] > 0.7)

# Save the merged file
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved as {output_path}")

