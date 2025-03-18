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

# Clean critic_score data for later use
df_with_scores = df.dropna(subset=['critic_score'])

# Console comparison - top platforms by total sales
plt.figure(figsize=(14, 8))
console_sales = df.groupby('console')['total_sales'].sum().sort_values(ascending=False).head(10)

ax = sns.barplot(x=console_sales.values, y=console_sales.index)
plt.title('Top 10 Gaming Platforms by Total Sales', fontsize=14)
plt.xlabel('Total Sales (millions)')
plt.ylabel('Console')

# Add value labels
for i, v in enumerate(console_sales.values):
    ax.text(v + 0.5, i, f'{v:.1f}', va='center')

plt.tight_layout()
plt.savefig('console_sales.png')
plt.show()

# Genre popularity by platform - focusing on top 3 platforms
top3_platforms = console_sales.head(3).index
platform_genre = df[df['console'].isin(top3_platforms)].pivot_table(
    index='genre',
    columns='console', 
    values='total_sales', 
    aggfunc='sum',
    fill_value=0
)

# Normalize to show percentage of platform's total sales
platform_genre_pct = platform_genre.div(platform_genre.sum(), axis=1) * 100

# Plot stacked bar chart for top genres on top platforms
top_genres = df.groupby('genre')['total_sales'].sum().nlargest(5).index
plt.figure(figsize=(14, 10))
platform_genre_pct.loc[top_genres].plot(kind='bar')
plt.title('Top 5 Genres by Platform (% of Platform Sales)', fontsize=14)
plt.xlabel('Genre')
plt.ylabel('Percentage of Platform Sales')
plt.legend(title='Platform')
plt.tight_layout()
plt.savefig('platform_genre_distribution.png')
plt.show()

# Average critic scores by platform (for platforms with sufficient data)
plt.figure(figsize=(14, 8))
# Calculate number of games with critic scores per platform
platform_counts = df_with_scores.groupby('console').size()
# Filter for platforms with at least 10 games that have critic scores
platforms_with_enough_data = platform_counts[platform_counts >= 10].index.tolist()

# Filter dataframe for those platforms only
filtered_df = df_with_scores[df_with_scores['console'].isin(platforms_with_enough_data)]
platform_scores = filtered_df.groupby('console')['critic_score'].mean().sort_values(ascending=False)

ax = sns.barplot(x=platform_scores.values, y=platform_scores.index)
plt.title('Average Critic Scores by Platform (Platforms with 10+ rated games)', fontsize=14)
plt.xlabel('Average Critic Score')
plt.ylabel('Platform')

# Add value labels
for i, v in enumerate(platform_scores.values):
    ax.text(v + 0.05, i, f'{v:.2f}', va='center')

plt.tight_layout()
plt.savefig('platform_critic_scores.png')
plt.show() 