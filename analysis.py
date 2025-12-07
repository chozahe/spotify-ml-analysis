# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully!")

# %%
df = pd.read_csv('data/spotify_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()

# %%
print("Data types:")
df.dtypes

# %%
print("Missing values:")
df.isnull().sum()

# %%
print("Basic statistics:")
df.describe()

# %%
print(f"Duplicates: {df.duplicated().sum()}")

# %%
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print("Distributions plotted!")

# %%
plt.figure(figsize=(10, 8))
correlation = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png', dpi=100, bbox_inches='tight')
plt.show()
print("Correlation matrix done!")

# %%
print("Basic EDA completed! Ready for modeling.")
