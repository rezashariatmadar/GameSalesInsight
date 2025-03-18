# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv('vgchartz-2024.csv')  # Update filename if different

# Display info about numerical columns
numerical_cols = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']
print("\nNumerical Columns Info:")
print(df[numerical_cols].info())
print("\nNumerical Columns Summary Statistics:")
print(df[numerical_cols].describe())

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

# Additional violin plots for sales distribution
plt.figure(figsize=(12, 6))
sns.violinplot(data=sales_data, x='Region', y='Sales')
plt.title('Violin Plot of Sales Distribution by Region')
plt.xticks(rotation=45)
plt.xlabel('Region')
plt.ylabel('Sales (millions)')
plt.tight_layout()
plt.savefig('sales_violin_plot.png')
plt.show()

# Correlation heatmap of numerical variables
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Sales and Critic Scores')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

