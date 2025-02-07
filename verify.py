import pandas as pd
import os

def concatenate_texts_by_lens_id(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(os.path.join('dataset', input_file))
    
    # Print column names and their string representations to debug
    print(f"Columns in {input_file}:", df.columns.tolist())
    print("Column names with repr():", [repr(col) for col in df.columns])
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Group by lens_id and concatenate texts
    concatenated_df = df.groupby('lens_id')['text'].apply(' '.join).reset_index()
    
    # Save the concatenated results
    concatenated_df.to_csv(os.path.join('dataset', output_file), index=False)
    print(f"Created {output_file} successfully!")

def main():
    # Process true_set.csv
    concatenate_texts_by_lens_id('true_set.csv', 'conc_true_set.csv')
    
    # Process new_texts.csv
    concatenate_texts_by_lens_id('new_texts.csv', 'conc_new_texts.csv')

if __name__ == "__main__":
    main()