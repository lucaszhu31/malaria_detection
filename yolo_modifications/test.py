import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the global font family to Prima Sans Roman
# Note: The font name must match the exact name installed on your OS
plt.rcParams['font.family'] = 'Prima Sans Roman'

# Data from your uploaded confusion matrix
data = [
    [5030, 1, 712],  # Predicted Trophozoite
    [0, 2662, 347],   # Predicted WBC
    [956, 433, 0]     # Predicted background
]

columns = ['Trophozoite', 'WBC', 'background']
df_cm = pd.DataFrame(data, index=columns, columns=columns)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 9))

# Create heatmap matching your blue color scheme
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            annot_kws={"size": 10}) # You can set font size for the numbers here

# Formatting labels
plt.title('Confusion Matrix\n', fontsize=12)
plt.xlabel('True', fontsize=12)
plt.ylabel('Predicted', fontsize=12)

# Specific font adjustments for tick labels
ax.set_xticklabels(columns, rotation=90)
ax.set_yticklabels(columns, rotation=0)

plt.tight_layout()
plt.show()