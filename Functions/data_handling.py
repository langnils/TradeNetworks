def load_network_dataframes(csv_filenames):
    """
    Loads network dataframes from a list of CSV filenames.
    Returns a dictionary with keys as dataframe names and values as DataFrames.
    """
    import os
    import pandas as pd
    W_ij_dataframes = {}
    print("\nLoading W_ij Dataframes from CSV files...")
    for filename in csv_filenames:
        try:
            df_reloaded = pd.read_csv(filename)
            base_name = os.path.basename(filename) # Gets 'W_1_ij.csv'
            parts = base_name.split('_') # Splits into ['W', '1', 'ij.csv']
            # Check if the parts list has at least 2 elements and the second part is a number
            if len(parts) >= 2 and parts[1].isdigit():
                df_number = parts[1]
                new_df_name = f"W_{df_number}_ij"
            else:
                # Fallback if filename format is unexpected, or use the full name as before
                new_df_name = os.path.splitext(base_name)[0]
            W_ij_dataframes[new_df_name] = df_reloaded
            print(f"Loaded {filename} as {new_df_name}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return W_ij_dataframes

def filter_dataframes_by_period(W_ij_dataframes, start_period, end_period):
    """
    Filters each DataFrame in the dictionary for the given period range.
    Returns a new dictionary with filtered DataFrames.
    """
    print(f"\nFiltering DataFrames for periods between {start_period} and {end_period}...")
    filtered_dataframes = {}
    for df_name, df_content in W_ij_dataframes.items():
        if not df_content.empty and 'period' in df_content.columns:
            original_rows = len(df_content)
            filtered_df = df_content[(df_content['period'] >= start_period) & (df_content['period'] <= end_period)]
            filtered_dataframes[df_name] = filtered_df
            print(f"Filtered {df_name}. Original rows: {original_rows}, Filtered rows: {len(filtered_df)}")
        else:
            filtered_dataframes[df_name] = df_content
            print(f"Skipping filtering for {df_name}: DataFrame is empty or 'period' column is missing.")
    return filtered_dataframes