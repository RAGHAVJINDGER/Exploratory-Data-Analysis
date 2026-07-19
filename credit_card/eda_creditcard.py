import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load & inspect the data
df = pd.read_csv("credit_card.csv")

print("First 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nInfo:")
df.info()

# 2. Detect missing / invalid values
df.replace('?', pd.NA, inplace=True)

print("\nMissing values per column (before cleaning):")
print(df.isnull().sum())


# 3. Handle missing values

for col in ['Personal Loan', 'Securities Account', 'CD Account']:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values per column (after cleaning):")
print(df.isnull().sum())


# 4. Fix irregular / outlier values

if 'Experience' in df.columns:
    print("\nRows with negative Experience (before fix):", (df['Experience'] < 0).sum())
    df['Experience'] = df['Experience'].abs()
    print("Rows with negative Experience (after fix):", (df['Experience'] < 0).sum())


# 5. Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())


# 6. Value counts

print("\nEducation value counts:")
print(df['Education'].value_counts())

print("\nFamily size value counts (normalized):")
print(df['Family'].value_counts(normalize=True))


# 7. Univariate analysis: histogram, skewness, kurtosis

plt.figure(figsize=(7, 5))
sns.histplot(df['Income'], kde=True)
plt.title("Distribution of Income")
plt.xlabel("Income")
plt.savefig("income_distribution.png")
plt.close()

print("\nIncome Skewness:", df['Income'].skew())
print("Income Kurtosis:", df['Income'].kurt())


# 8. Bivariate analysis: numeric vs numeric
plt.figure(figsize=(7, 5))
plt.scatter(df['Income'], df['CCAvg'], alpha=0.4)
plt.xlabel("Income")
plt.ylabel("Credit Card Average Spend (CCAvg)")
plt.title("Income vs CCAvg")
plt.savefig("income_vs_ccavg.png")
plt.close()


# 9. Bivariate analysis: numeric vs categorical

plt.figure(figsize=(8, 5))
sns.boxplot(x='Education', y='Income', data=df)
plt.title("Income by Education Level")
plt.savefig("income_by_education.png")
plt.close()

print("\nAverage Income by Personal Loan status:")
print(df.groupby('Personal Loan')['Income'].mean())


# 10. Correlation matrix / heatmap

corr_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
corr = df[corr_cols].corr()

print("\nCorrelation matrix:")
print(corr)

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("\nDone! Plots saved as PNG files in the current folder:")
print(" - income_distribution.png")
print(" - income_vs_ccavg.png")
print(" - income_by_education.png")
print(" - correlation_heatmap.png")