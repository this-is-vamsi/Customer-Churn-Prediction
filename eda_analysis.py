"""
Customer Churn Prediction - Exploratory Data Analysis
This script performs comprehensive EDA including:
- Data Cleaning
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("="*80)
print("CUSTOMER CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

df = pd.read_csv('data/churn.csv')
print("\n1. DATA LOADING COMPLETE")
print(f"Dataset Shape: {df.shape}")
print(f"Total Customers: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("2. DATA CLEANING")
print("="*80)

# 2.1 Display basic information
print("\n2.1 Dataset Information:")
print(df.info())

print("\n2.2 First few rows:")
print(df.head())

print("\n2.3 Statistical Summary:")
print(df.describe())

# 2.4 Check for missing values
print("\n2.4 Missing Values:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Percentage': missing_percentage
})
print(missing_df[missing_df['Missing_Count'] > 0])

if missing_df['Missing_Count'].sum() == 0:
    print("No missing values found!")

# 2.5 Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n2.5 Duplicate Rows: {duplicates}")

# 2.6 Check data types
print("\n2.6 Data Types:")
print(df.dtypes)

# 2.7 Drop unnecessary columns (RowNumber, CustomerId, Surname)
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print(f"\n2.7 Columns dropped: RowNumber, CustomerId, Surname")
print(f"Updated shape: {df_clean.shape}")

# 2.8 Check for outliers using IQR method
print("\n2.8 Outlier Detection (IQR Method):")
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers ({(outliers/len(df_clean)*100):.2f}%)")

# ============================================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. UNIVARIATE ANALYSIS")
print("="*80)

# 3.1 Target Variable Distribution
print("\n3.1 Target Variable (Exited) Distribution:")
churn_counts = df_clean['Exited'].value_counts()
churn_percentage = df_clean['Exited'].value_counts(normalize=True) * 100
print(f"Not Churned (0): {churn_counts[0]} ({churn_percentage[0]:.2f}%)")
print(f"Churned (1): {churn_counts[1]} ({churn_percentage[1]:.2f}%)")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df_clean, x='Exited', ax=axes[0], palette='Set2')
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Exited (0=No, 1=Yes)')
axes[0].set_ylabel('Count')

axes[1].pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', 
            colors=['#66b3ff', '#ff9999'], startangle=90)
axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('univariate_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: univariate_target_distribution.png")

# 3.2 Numerical Features Distribution
print("\n3.2 Numerical Features Distribution:")
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                      'NumOfProducts', 'EstimatedSalary']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df_clean[col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    
    # Add statistics
    mean_val = df_clean[col].mean()
    median_val = df_clean[col].median()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('univariate_numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: univariate_numerical_distributions.png")

# Print statistics for numerical features
print("\nNumerical Features Statistics:")
for col in numerical_features:
    print(f"\n{col}:")
    print(f"  Mean: {df_clean[col].mean():.2f}")
    print(f"  Median: {df_clean[col].median():.2f}")
    print(f"  Std Dev: {df_clean[col].std():.2f}")
    print(f"  Min: {df_clean[col].min():.2f}")
    print(f"  Max: {df_clean[col].max():.2f}")
    print(f"  Skewness: {df_clean[col].skew():.2f}")
    print(f"  Kurtosis: {df_clean[col].kurtosis():.2f}")

# 3.3 Categorical Features Distribution
print("\n3.3 Categorical Features Distribution:")
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    value_counts = df_clean[col].value_counts()
    sns.countplot(data=df_clean, x=col, ax=axes[idx], palette='Set3')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    
    # Add percentage labels
    total = len(df_clean)
    for p in axes[idx].patches:
        height = p.get_height()
        axes[idx].text(p.get_x() + p.get_width()/2., height + 50,
                      f'{height/total*100:.1f}%',
                      ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('univariate_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: univariate_categorical_distributions.png")

# Print categorical feature statistics
for col in categorical_features:
    print(f"\n{col}:")
    print(df_clean[col].value_counts())
    print(f"Percentage Distribution:")
    print(df_clean[col].value_counts(normalize=True) * 100)

# 3.4 Box plots for outlier visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    sns.boxplot(data=df_clean, y=col, ax=axes[idx], color='lightblue')
    axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig('univariate_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: univariate_boxplots.png")

# ============================================================================
# 4. BIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. BIVARIATE ANALYSIS")
print("="*80)

# 4.1 Numerical Features vs Churn
print("\n4.1 Numerical Features vs Churn:")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    # Box plot
    sns.boxplot(data=df_clean, x='Exited', y=col, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{col} vs Churn', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Exited (0=No, 1=Yes)')
    axes[idx].set_ylabel(col)
    
    # Add mean values
    means = df_clean.groupby('Exited')[col].mean()
    for i, mean in enumerate(means):
        axes[idx].text(i, mean, f'{mean:.2f}', ha='center', va='bottom', 
                      fontweight='bold', color='red', fontsize=10)

plt.tight_layout()
plt.savefig('bivariate_numerical_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bivariate_numerical_vs_churn.png")

# Statistical tests for numerical features
print("\nStatistical Tests (T-test for numerical features vs Churn):")
for col in numerical_features:
    churned = df_clean[df_clean['Exited'] == 1][col]
    not_churned = df_clean[df_clean['Exited'] == 0][col]
    t_stat, p_value = stats.ttest_ind(churned, not_churned)
    print(f"{col}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
    if p_value < 0.05:
        print(f"  -> Significant difference (p < 0.05)")
    else:
        print(f"  -> No significant difference (p >= 0.05)")

# 4.2 Categorical Features vs Churn
print("\n4.2 Categorical Features vs Churn:")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    # Stacked bar chart
    churn_crosstab = pd.crosstab(df_clean[col], df_clean['Exited'], normalize='index') * 100
    churn_crosstab.plot(kind='bar', stacked=False, ax=axes[idx], color=['#66b3ff', '#ff9999'])
    axes[idx].set_title(f'{col} vs Churn Rate', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Percentage (%)')
    axes[idx].legend(['Not Churned', 'Churned'], title='Exited')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('bivariate_categorical_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bivariate_categorical_vs_churn.png")

# Chi-square tests for categorical features
print("\nChi-Square Tests (Categorical features vs Churn):")
for col in categorical_features:
    contingency_table = pd.crosstab(df_clean[col], df_clean['Exited'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"{col}: Chi2 = {chi2:.4f}, p-value = {p_value:.4e}")
    if p_value < 0.05:
        print(f"  -> Significant association (p < 0.05)")
    else:
        print(f"  -> No significant association (p >= 0.05)")

# 4.3 Churn rate by categorical features
print("\n4.3 Churn Rate by Categorical Features:")
for col in categorical_features:
    churn_rate = df_clean.groupby(col)['Exited'].mean() * 100
    print(f"\n{col} Churn Rate:")
    print(churn_rate)

# 4.4 Correlation between numerical features
print("\n4.4 Correlation Analysis:")
correlation_matrix = df_clean[numerical_features + ['Exited']].corr()
print("\nCorrelation with Churn (Exited):")
print(correlation_matrix['Exited'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('bivariate_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bivariate_correlation_heatmap.png")

# 4.5 Scatter plots for key relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age vs Balance
sns.scatterplot(data=df_clean, x='Age', y='Balance', hue='Exited', 
                palette='Set1', alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title('Age vs Balance (Colored by Churn)', fontsize=12, fontweight='bold')

# CreditScore vs EstimatedSalary
sns.scatterplot(data=df_clean, x='CreditScore', y='EstimatedSalary', hue='Exited', 
                palette='Set1', alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('CreditScore vs EstimatedSalary (Colored by Churn)', 
                     fontsize=12, fontweight='bold')

# Age vs CreditScore
sns.scatterplot(data=df_clean, x='Age', y='CreditScore', hue='Exited', 
                palette='Set1', alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title('Age vs CreditScore (Colored by Churn)', fontsize=12, fontweight='bold')

# Tenure vs NumOfProducts
sns.scatterplot(data=df_clean, x='Tenure', y='NumOfProducts', hue='Exited', 
                palette='Set1', alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('Tenure vs NumOfProducts (Colored by Churn)', 
                     fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('bivariate_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bivariate_scatter_plots.png")

# ============================================================================
# 5. MULTIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. MULTIVARIATE ANALYSIS")
print("="*80)

# 5.1 Pair plot for key features
print("\n5.1 Creating Pair Plot...")
key_features = ['Age', 'Balance', 'NumOfProducts', 'CreditScore', 'Exited']
pair_plot = sns.pairplot(df_clean[key_features], hue='Exited', palette='Set1', 
                          diag_kind='kde', plot_kws={'alpha': 0.6})
pair_plot.fig.suptitle('Pair Plot - Key Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('multivariate_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_pairplot.png")

# 5.2 Churn rate by Geography and Gender
print("\n5.2 Churn Rate by Geography and Gender:")
geo_gender_churn = df_clean.groupby(['Geography', 'Gender'])['Exited'].mean() * 100
print(geo_gender_churn)

fig, ax = plt.subplots(figsize=(10, 6))
geo_gender_churn.unstack().plot(kind='bar', ax=ax, color=['#66b3ff', '#ff9999'])
ax.set_title('Churn Rate by Geography and Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Geography')
ax.set_ylabel('Churn Rate (%)')
ax.legend(title='Gender')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('multivariate_geography_gender_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_geography_gender_churn.png")

# 5.3 Age groups analysis
print("\n5.3 Age Group Analysis:")
df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0, 30, 40, 50, 60, 100], 
                                labels=['<30', '30-40', '40-50', '50-60', '60+'])
age_group_analysis = df_clean.groupby('AgeGroup').agg({
    'Exited': ['count', 'sum', 'mean']
}).round(4)
age_group_analysis.columns = ['Total', 'Churned', 'Churn_Rate']
age_group_analysis['Churn_Rate'] = age_group_analysis['Churn_Rate'] * 100
print(age_group_analysis)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
age_group_analysis['Churn_Rate'].plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Churn Rate by Age Group', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].tick_params(axis='x', rotation=45)

age_gender_churn = df_clean.groupby(['AgeGroup', 'Gender'])['Exited'].mean() * 100
age_gender_churn.unstack().plot(kind='bar', ax=axes[1], color=['#66b3ff', '#ff9999'])
axes[1].set_title('Churn Rate by Age Group and Gender', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].legend(title='Gender')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('multivariate_age_group_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_age_group_analysis.png")

# 5.4 Product and Activity Analysis
print("\n5.4 Product and Activity Analysis:")
product_activity_churn = df_clean.groupby(['NumOfProducts', 'IsActiveMember'])['Exited'].agg(['count', 'mean'])
product_activity_churn.columns = ['Count', 'Churn_Rate']
product_activity_churn['Churn_Rate'] = product_activity_churn['Churn_Rate'] * 100
print(product_activity_churn)

fig, ax = plt.subplots(figsize=(10, 6))
product_activity_churn['Churn_Rate'].unstack().plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff'])
ax.set_title('Churn Rate by Number of Products and Active Membership', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Products')
ax.set_ylabel('Churn Rate (%)')
ax.legend(['Not Active', 'Active'], title='IsActiveMember')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('multivariate_product_activity_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_product_activity_analysis.png")

# 5.5 Geography, Gender, and Active Member Analysis
print("\n5.5 Geography, Gender, and Active Member Analysis:")
multi_analysis = df_clean.groupby(['Geography', 'Gender', 'IsActiveMember'])['Exited'].mean() * 100
print(multi_analysis)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
countries = df_clean['Geography'].unique()

for idx, country in enumerate(countries):
    country_data = df_clean[df_clean['Geography'] == country]
    cross_tab = pd.crosstab(country_data['Gender'], 
                            country_data['IsActiveMember'], 
                            values=country_data['Exited'], 
                            aggfunc='mean') * 100
    
    cross_tab.plot(kind='bar', ax=axes[idx], color=['#ff9999', '#66b3ff'])
    axes[idx].set_title(f'Churn Rate in {country}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Gender')
    axes[idx].set_ylabel('Churn Rate (%)')
    axes[idx].legend(['Not Active', 'Active'], title='IsActiveMember')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('multivariate_geography_gender_activity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_geography_gender_activity.png")

# 5.6 Balance distribution by Geography and Churn
print("\n5.6 Balance Distribution by Geography and Churn:")
fig, ax = plt.subplots(figsize=(12, 6))
for country in df_clean['Geography'].unique():
    for churn in [0, 1]:
        data = df_clean[(df_clean['Geography'] == country) & (df_clean['Exited'] == churn)]['Balance']
        label = f"{country} - {'Churned' if churn == 1 else 'Not Churned'}"
        ax.hist(data, bins=30, alpha=0.5, label=label)

ax.set_title('Balance Distribution by Geography and Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Balance')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('multivariate_balance_geography_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_balance_geography_churn.png")

# 5.7 3D visualization (Age, Balance, CreditScore)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

churned = df_clean[df_clean['Exited'] == 1]
not_churned = df_clean[df_clean['Exited'] == 0]

# Sample data for better visualization (if dataset is large)
sample_size = min(1000, len(churned), len(not_churned))
churned_sample = churned.sample(n=sample_size, random_state=42)
not_churned_sample = not_churned.sample(n=sample_size, random_state=42)

ax.scatter(not_churned_sample['Age'], not_churned_sample['Balance'], 
           not_churned_sample['CreditScore'], c='blue', marker='o', 
           alpha=0.5, label='Not Churned')
ax.scatter(churned_sample['Age'], churned_sample['Balance'], 
           churned_sample['CreditScore'], c='red', marker='^', 
           alpha=0.5, label='Churned')

ax.set_xlabel('Age')
ax.set_ylabel('Balance')
ax.set_zlabel('Credit Score')
ax.set_title('3D Scatter: Age vs Balance vs CreditScore', fontsize=14, fontweight='bold')
ax.legend()

plt.savefig('multivariate_3d_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: multivariate_3d_scatter.png")

# ============================================================================
# 6. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("6. KEY INSIGHTS SUMMARY")
print("="*80)

print(f"""
DATA OVERVIEW:
- Total Customers: {len(df_clean)}
- Churned Customers: {df_clean['Exited'].sum()} ({df_clean['Exited'].mean()*100:.2f}%)
- Non-Churned Customers: {len(df_clean) - df_clean['Exited'].sum()} ({(1-df_clean['Exited'].mean())*100:.2f}%)

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum()}
- Duplicate Rows: {duplicates}
- Columns Dropped: RowNumber, CustomerId, Surname (not useful for analysis)

KEY FINDINGS:

1. AGE:
   - Average age of churned customers: {df_clean[df_clean['Exited']==1]['Age'].mean():.2f} years
   - Average age of non-churned customers: {df_clean[df_clean['Exited']==0]['Age'].mean():.2f} years
   - Older customers show higher churn rates

2. GEOGRAPHY:
   - Germany has the highest churn rate: {df_clean[df_clean['Geography']=='Germany']['Exited'].mean()*100:.2f}%
   - France: {df_clean[df_clean['Geography']=='France']['Exited'].mean()*100:.2f}%
   - Spain: {df_clean[df_clean['Geography']=='Spain']['Exited'].mean()*100:.2f}%

3. GENDER:
   - Female churn rate: {df_clean[df_clean['Gender']=='Female']['Exited'].mean()*100:.2f}%
   - Male churn rate: {df_clean[df_clean['Gender']=='Male']['Exited'].mean()*100:.2f}%

4. ACTIVE MEMBERSHIP:
   - Inactive members churn rate: {df_clean[df_clean['IsActiveMember']==0]['Exited'].mean()*100:.2f}%
   - Active members churn rate: {df_clean[df_clean['IsActiveMember']==1]['Exited'].mean()*100:.2f}%

5. NUMBER OF PRODUCTS:
   - Customers with 1 product: {df_clean[df_clean['NumOfProducts']==1]['Exited'].mean()*100:.2f}% churn rate
   - Customers with 2 products: {df_clean[df_clean['NumOfProducts']==2]['Exited'].mean()*100:.2f}% churn rate
   - Customers with 3+ products: {df_clean[df_clean['NumOfProducts']>=3]['Exited'].mean()*100:.2f}% churn rate

6. BALANCE:
   - Average balance of churned: ${df_clean[df_clean['Exited']==1]['Balance'].mean():,.2f}
   - Average balance of non-churned: ${df_clean[df_clean['Exited']==0]['Balance'].mean():,.2f}

All visualizations have been saved as PNG files in the current directory.
""")

print("\n" + "="*80)
print("EDA COMPLETE! All visualizations and analysis saved.")
print("="*80)

# Save cleaned dataset
df_clean.to_csv('data/churn_cleaned.csv', index=False)
print("\nCleaned dataset saved as: data/churn_cleaned.csv")
