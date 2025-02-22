import pandas as pd

# Load the CSV file
file_path = "dataset/new_test_set.csv"
df = pd.read_csv(file_path)

# Concatenate text column based on lens_id
df_grouped = df.groupby("lens_id", as_index=False)[" text"].agg(" ".join)

# Save the updated CSV file
df_grouped.to_csv(file_path, index=False)

print("File updated and saved successfully!")
