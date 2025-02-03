import csv
import sys
import os

# Increase the field size limit
csv.field_size_limit(sys.maxsize)

def detailed_csv_analysis(file_paths):
    """
    Provide detailed analysis of CSV files
    
    Args:
        file_paths (list): List of paths to CSV files
    
    Returns:
        dict: Detailed information about each CSV file
    """
    file_details = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                # Use csv reader to properly handle CSV files
                csv_reader = csv.reader(csvfile)
                
                # Read header
                header = next(csv_reader, None)
                
                # Count remaining lines (documents)
                document_count = sum(1 for _ in csv_reader)
                
                # Reset file pointer to check file details again
                csvfile.seek(0)
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                
                # Get sample text length
                first_row = next(csv_reader, None)
                sample_text_length = len(first_row[1]) if first_row and len(first_row) > 1 else 0
                
                # Get file size
                file_size = os.path.getsize(file_path)
                
                file_details[file_path] = {
                    'total_lines': document_count + 1,  # +1 to include header
                    'document_count': document_count,
                    'headers': header,
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size / (1024 * 1024),
                    'sample_text_length': sample_text_length
                }
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return file_details

if __name__ == "__main__":
    # File paths
    csv_files = [
        'data/true_set/true_set.csv',
        'data/new_texts/new_texts.csv'
    ]
    
    results = detailed_csv_analysis(csv_files)
    
    # Print detailed results
    for file, details in results.items():
        print(f"\nFile: {file}")
        for key, value in details.items():
            print(f"{key}: {value}")