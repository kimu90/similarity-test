import pandas as pd

# File paths
result_path = "dataset/result.csv"
new_true_set_path = "dataset/new_true_set.csv"

# Load the CSV files
df_result = pd.read_csv(result_path)
df_new_true = pd.read_csv(new_true_set_path)

# Ensure 'lens_id' column is of the same type
df_result["lens_id"] = df_result["lens_id"].astype(str)
df_new_true["lens_id"] = df_new_true["lens_id"].astype(str)

# Concatenate text based on 'lens_id'
df_new_true_grouped = df_new_true.groupby("lens_id")["text"].apply(lambda x: " ".join(x.astype(str))).reset_index()

# Merge with result.csv
df_result = df_result.merge(df_new_true_grouped, on="lens_id", how="left")

# Save the updated result.csv
df_result.to_csv(result_path, index=False)

print("File updated and saved successfully!")
