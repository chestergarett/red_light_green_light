import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON file
with open(r"outputs/player_metrics.json", "r") as file:
    data = json.load(file)

# Convert JSON to a DataFrame
df = pd.DataFrame(data)

# List of metrics to visualize
metrics = ["total_distance_meters", "ave_speed_m_s", "stop_count", 
           "move_count", "motion_rate_variation", "distance_per_motion"]

# Create a bar graph for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["player_id"], df[metric], color="skyblue")
    plt.xlabel("Player ID")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Comparison of {metric.replace('_', ' ').title()} per Player")
    # Set the x-axis to categorical values
    player_ids = df["player_id"].astype(int)  # Ensure Player IDs are integers
    plt.xticks(ticks=player_ids, labels=player_ids, rotation=0)  # Set ticks and labels
    for bar in bars:
        yval = bar.get_height()  # Get the height of each bar (the value)
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
            yval,  # Y position (top of the bar)
            f'{yval:.2f}',  # Display the value formatted to two decimal places
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment (place text above the bar)
            fontsize=10,  # Font size of the labels
            color='black'  # Color of the text
        )
    
    plt.tight_layout()
    filename = f"outputs/metrics/{metric}.jpg"  # Name the file based on the metric
    plt.savefig(filename, format="jpg", dpi=300)  # Save with high resolution
    
    # Close the plot to free memory
    plt.close()