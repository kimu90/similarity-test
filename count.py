import sys
import csv

# Increase the field size limit
csv.field_size_limit(sys.maxsize)

def count_csv_lines(file_paths):
    """
    Count the number of lines in CSV files, excluding the header.
    
    Args:
        file_paths (list): List of paths to CSV files
    
    Returns:
        dict: A dictionary with file names and their line counts
    """
    line_counts = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                # Use csv reader to properly handle CSV files
                csv_reader = csv.reader(csvfile)
                
                # Skip header
                next(csv_reader, None)
                
                # Count remaining lines
                line_count = sum(1 for _ in csv_reader)
                
                line_counts[file_path] = line_count
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return line_counts

if __name__ == "__main__":
    # File paths
    csv_files = [
        'data/true_set/true_set.csv',
        'data/new_texts/new_texts.csv'
    ]
    
    results = count_csv_lines(csv_files)
    
    # Print results
    for file, count in results.items():
        print(f"{file}: {count} lines")