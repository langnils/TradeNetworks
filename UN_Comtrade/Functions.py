#Library used in the functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Pandas Manipulation Help - I know.... 

# Function to drop columns by their index
def drop_columns_by_index(df, indices):
    columns_to_drop = [df.columns[i] for i in indices]
    df_dropped = df.drop(columns=columns_to_drop)
    return df_dropped


# Function to print column names and their index
def print_columns(df):
    print("Column names and their index:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")
        print(f'{col}: \nTotal Unique Values: {df[col].nunique()}, Top Value: {df[col].mode().iloc[0]}, Frequency of Top Value: {df[col].value_counts().iloc[0]}, Data Type: {df[col].dtype} \n\n')
    

# Function to drop rows by column index and value
def drop_rows_by_index_value(df, col_index, value):
    column_name = df.columns[col_index]
    df_dropped = df[df[column_name] != value]
    return df_dropped


# Function to get all unique values and their frequencies in a specific column by index
def get_values_and_frequencies(df, col_index):
    column_name = df.columns[col_index]
    value_counts = df[column_name].value_counts()
    return value_counts

# Function to drop empty or all-NA columns and print them
def clean_dataframe(df):
    # Get the name of the DataFrame variable
    df_name = [name for name, value in locals().items() if value is df][0]
    
    # Identify columns that are all NaN
    empty_columns = df.columns[df.isna().all()].tolist()
    if empty_columns:
        print(f"Dropping columns from {df_name}: {empty_columns}")
    # Drop empty columns
    return df.dropna(axis=1, how='all')

# Function to fix datatypes in a DataFrame
def fix_datatypes(df):
    """
    Fixes the data types of the DataFrame by converting yearly data (both
    string and integer) to datetime only if they match the format of a year,
    and other columns to numeric types where applicable.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with corrected data types.
    """
    # Get the name of the DataFrame variable (if needed)
    df_name = [name for name, value in locals().items() if value is df][0]

    # Convert columns to datetime if they represent years
    for column in df.columns:
        try:
            # Check for four-digit year format in string columns
            if df[column].dtype == 'object' and df[column].str.match(r'^\d{4}$').all():
                df[column] = pd.to_datetime(df[column], format='%Y')
                print(f"Converted {column} (string) to datetime format.")
            # Check for integer columns that represent years
            elif df[column].dtype == 'int64' and df[column].between(1900, 2100).all():
                df[column] = pd.to_datetime(df[column], format='%Y')
                print(f"Converted {column} (int) to datetime format.")
            else:
                # Attempt to convert to numeric if it's not datetime
                df[column] = pd.to_numeric(df[column], errors='coerce')
                if df[column].isnull().any():
                    print(f"Converted {column} to numeric format with some NaNs.")
        except Exception as e:
            print(f"Could not convert {column}: {e}")
    
    return df



#Datapreparation Functions

#Generate a year rang as a list
def generate_year_range_string(start_year, end_year):
    years = list(range(start_year, end_year + 1))
    year_strings = [str(year) for year in years]
    year_range_string = ",".join(year_strings)
    return year_range_string

def generate_comma_separated_string(items):
    """
    Generate a comma-separated string from a list of items.
    
    Parameters:
    items (list): A list of items to be converted to a comma-separated string.
    
    Returns:
    str: A comma-separated string of the items.
    """
    return ",".join(items)




# Visualizing in Python


def plot_bar_chart(df, col1, col2):
    """
    Creates a bar chart based on two columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the column for the x-axis.
    col2 (str): The name of the column for the y-axis.
    """
    # Ensure the columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}' and '{col2}' do not exist in the DataFrame.")
    
    # Check if col1 is datetime and convert it to a string format for plotting
    if pd.api.types.is_datetime64_any_dtype(df[col1]):
        df[col1] = df[col1].dt.strftime('%Y')  # Convert to string representation of the year
        print("Info: Converted the datetime to yearly data strings")
    
    # Group by col1 and sum the values in col2
    data = df.groupby(col1)[col2].sum().reset_index()
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(data[col1], data[col2], color='skyblue')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Bar Chart of {col2} by {col1}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()




def draw_trade_graph(edges, title): 

    print("Initializing Trade Graph")
    
        # Create a directed graph
    G = nx.DiGraph()


    #Add the edges to the graph
    for i in range(len(edges)):
        G.add_edge(edges["reporterISO"][i], edges["partnerISO"][i], weight = edges["primaryValue"][i])

    # Calculate the sum of incoming edge weights for each node
    node_weight_sum = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes()}

    # Identify the top 10 nodes based on the sum of incoming edge weights
    top_10_nodes = sorted(node_weight_sum, key=node_weight_sum.get, reverse=True)[:10]
    other_nodes = [node for node in G.nodes if node not in top_10_nodes]

    # Sort the other nodes based on their weight sums in descending order
    other_nodes_sorted = sorted(other_nodes, key=node_weight_sum.get, reverse=True)
    top_10_nodes_sorted = sorted(top_10_nodes, key=lambda n: G.degree(n), reverse=True)

    # Generate positions for the top 10 nodes (placed in a circle)
    top_10_pos = {}
    angle_step = 2 * np.pi / len(top_10_nodes_sorted)
    for i, node in enumerate(top_10_nodes):
        angle = i * angle_step
        top_10_pos[node] = (np.cos(angle), np.sin(angle))

    # Generate positions for the remaining nodes (placed in a larger circle, sorted clockwise by degree)
    other_pos = {}
    angle_step = 2 * np.pi / len(other_nodes_sorted)
    for i, node in enumerate(other_nodes_sorted):
        angle = i * angle_step
        other_pos[node] = (2 * np.cos(angle), 2 * np.sin(angle))

    # Combine positions
    pos = {**top_10_pos, **other_pos}

    # Normalize edge weights for width
    weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    min_weight, max_weight = np.min(weights), np.max(weights)
    norm_weights = (weights - min_weight) / (max_weight - min_weight)
    edge_widths = 0.1 + 10 * norm_weights  # Scale widths between 0.1 and 5


    # Normalize node sizes based on the sum of incoming edge weights
    min_size, max_size = 100, 1000  # Min and max node size
    node_sizes = np.array([node_weight_sum[n] for n in G.nodes()])
    norm_node_sizes = min_size + (max_size - min_size) * (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min())

    # Define node colors, highlighting one specific country
    default_color = 'blue'
    highlight_color = 'red'
    node_colors = [highlight_color if node == 'CHN' or node == "HKG" else default_color for node in G.nodes()]

    # Set the figure size
    plt.figure(figsize=(12, 12))  # Adjust the figure size as needed

    # Draw nodes with size proportional to the sum of incoming edge weights
    nx.draw_networkx_nodes(G, pos, node_size=norm_node_sizes, node_color=node_colors, alpha=0.6)

    # Draw edges with normalized widths
    edge_width = dict(zip(G.edges, edge_widths))
    nx.draw_networkx_edges(G, pos, width=[edge_width[edge] for edge in G.edges()], edge_color='gray', alpha=0.5, connectionstyle="arc3,rad=0.4", arrowsize= 5)



    # Draw edge labels
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(title)
    plt.show()


    #TODO Analyze data with HONG kong and China combined and also take a look at Taiwan