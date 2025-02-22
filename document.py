import pandas as pd
import os

# Define file paths
data_folder = "dataset"
og_new_path = os.path.join(data_folder, "false_claim.csv")
csv_files = ["euclidean_similarity_results.csv"]

# Load og_new.csv
if not os.path.exists(og_new_path):
    print("og_new.csv not found in data folder.")
    exit()

og_df = pd.read_csv(og_new_path)

# Strip spaces from column names
og_df.columns = og_df.columns.str.strip()

# Print actual column names for debugging
print("Columns in og_new.csv:", list(og_df.columns))

# Ensure required columns exist
if "lens_id" not in og_df.columns or "text" not in og_df.columns:
    print(f"Missing required columns in og_new.csv. Found: {list(og_df.columns)}")
    exit()

# Process each CSV file in the data folder
for file in csv_files:
    file_path = os.path.join(data_folder, file)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Strip spaces from column names in the target CSV
        df.columns = df.columns.str.strip()

        # Ensure required column exists in the target CSV
        if "text_id" not in df.columns:
            print(f"Missing 'text_id' column in {file}. Found: {list(df.columns)}. Skipping.")
            continue

        # Merge based on matching text_id (from CSV) with lens_id (from og_new.csv)
        df = df.merge(og_df[["lens_id", "text"]], left_on="text_id", right_on="lens_id", how="left")

        # Drop the extra lens_id column after merging
        df.drop(columns=["lens_id"], inplace=True)

        # Save updated CSV in the same folder
        df.to_csv(file_path, index=False)
        print(f"Updated {file} saved in {data_folder}/")
    else:
        print(f"{file} not found. Skipping.")