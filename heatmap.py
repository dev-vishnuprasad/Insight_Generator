import pandas as pd
import seaborn as sns
from generate_plot import objToCat
import json
import matplotlib.pyplot as plt
import os

def heatmap(test_path,train_path,output_path):

    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)

    Processed_data = pd.concat([train_data,test_data])
    Processed_data = objToCat(Processed_data)
    
    #defining correlation between only numerical columns
    corr = Processed_data.select_dtypes(include=['number']).corr()
    base_filename = "heatmap_correlation"

    print("heatmap")
    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, 
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        annot=True,  
        cmap='coolwarm', 
        vmin=-1, vmax=1  
    )
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    # Save the heatmap image
    heatmap_filename = f"{base_filename}.png"
    plt.savefig(os.path.join('/output', heatmap_filename))
    plt.close()  # Close the plot to free up memory

    print(f"Heatmap saved as {heatmap_filename}")

    # Create JSON representation of the heatmap data
    heatmap_data = []
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            heatmap_data.append({
                'x': col,
                'y': row,
                'value': corr.iloc[i, j]
            })

    # Save the JSON to a file in output folder
    json_filename = f"{base_filename}.json"
    with open(os.path.join('/output', heatmap_filename), 'w') as f:
        json.dump(heatmap_data, f, indent=2)

    print(f"JSON data saved as {json_filename}")





if __name__ == "__main__":

    heatmap('preprocessed/preprocessed_test_data.csv', 'preprocessed/preprocessed_train_data.csv', 'output/')
