# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv('vgchartz-2024.csv')

# Display info about numerical columns
numerical_cols = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']

# Create a summary DataFrame for numerical columns
summary = pd.DataFrame({
    'Data Type': df[numerical_cols].dtypes,
    'Non-Null Count': df[numerical_cols].count(),
    'Null Count': df[numerical_cols].isnull().sum(),
    'Mean': df[numerical_cols].mean(),
    'Median': df[numerical_cols].median(),
    'Std Dev': df[numerical_cols].std(),
    'Min': df[numerical_cols].min(),
    'Max': df[numerical_cols].max()
}).round(2)

# Save summary to CSV
summary.to_csv('numerical_summary.csv')

# Create subplots for different visualization types
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Distribution Analysis of Video Game Sales', fontsize=16)

# 1. Histogram of total sales
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='total_sales', bins=30)
plt.title('Histogram of Total Sales')
plt.xlabel('Total Sales (millions)')
plt.ylabel('Count')

# 2. Box plot of sales by region
plt.subplot(2, 2, 2)
sales_data = pd.melt(df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']], 
                     var_name='Region', value_name='Sales')
sns.boxplot(data=sales_data, x='Region', y='Sales')
plt.title('Box Plot of Sales by Region')
plt.xticks(rotation=45)
plt.xlabel('Region')
plt.ylabel('Sales (millions)')

# 3. Q-Q plot of total sales
plt.subplot(2, 2, 3)
import scipy.stats as stats
stats.probplot(df['total_sales'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Total Sales')

# 4. Scatter plot of critic score vs total sales
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='critic_score', y='total_sales', alpha=0.5)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales (millions)')

plt.tight_layout()
plt.savefig('distribution_analysis.png')
plt.show()

# Print correlation matrix
correlation_matrix = df[numerical_cols].corr().round(3)
correlation_matrix.to_csv('correlation_matrix.csv') 