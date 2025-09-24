import pandas as pd

def remove_duplicate_entries(input_file_path, output_file_path):
    try:
      
        df = pd.read_csv(input_file_path)

        initial_row_count = len(df)
        print(f"Original number of rows: {initial_row_count}")

       
        df_cleaned = df.drop_duplicates(keep='first')

     
        final_row_count = len(df_cleaned)
        print(f"Number of rows after removing duplicates: {final_row_count}")

        duplicates_removed = initial_row_count - final_row_count
        print(f"Number of duplicate rows removed: {duplicates_removed}")

       
        df_cleaned.to_csv(output_file_path, index=False)
        print(f"\nCleaned data has been successfully saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    
    input_csv = 'Training_old.csv'
    output_csv = 'Training.csv'

    remove_duplicate_entries(input_csv, output_csv)
