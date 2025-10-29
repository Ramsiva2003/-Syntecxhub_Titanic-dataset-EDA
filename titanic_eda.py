# ------------------------------------------------------
# 🧭 Syntecxhub Internship | Project 1: Titanic Dataset EDA
# ------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------
# Step 1: Create Folder for Graphs
# ------------------------------------------------------
os.makedirs("graphs", exist_ok=True)

# Step 2: Load Dataset
df = sns.load_dataset("titanic")
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Step 3: Basic Info
print("\n📊 Data Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Handle Missing Values
df['age'].fillna(df['age'].median(), inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
df.drop(['deck'], axis=1, inplace=True)

# Step 5: Add Age Groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 30, 50, 80],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)

# ------------------------------------------------------
# Step 6: Visualizations (Show + Save)
# ------------------------------------------------------

# 1️⃣ Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sex', hue='survived', palette='coolwarm')
plt.title('Survival Count by Gender')
plt.xlabel('Gender'); plt.ylabel('Count')
plt.savefig("graphs/1_survival_by_gender.png", bbox_inches='tight')
plt.show()

# 2️⃣ Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='class', hue='survived', palette='Set2')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Class'); plt.ylabel('Count')
plt.savefig("graphs/2_survival_by_class.png", bbox_inches='tight')
plt.show()

# 3️⃣ Survival by Age Group
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', hue='survived', palette='magma')
plt.title('Survival Count by Age Group')
plt.xlabel('Age Group'); plt.ylabel('Count')
plt.savefig("graphs/3_survival_by_age.png", bbox_inches='tight')
plt.show()

# 4️⃣ Age Distribution by Class
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='class', y='age', palette='viridis')
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class'); plt.ylabel('Age')
plt.savefig("graphs/4_age_by_class.png", bbox_inches='tight')
plt.show()

# 5️⃣ Fare Distribution by Class and Survival
plt.figure(figsize=(8,5))
sns.violinplot(data=df, x='class', y='fare', hue='survived', split=True, palette='cool')
plt.title('Fare Distribution by Class and Survival')
plt.xlabel('Passenger Class'); plt.ylabel('Fare')
plt.savefig("graphs/5_fare_distribution.png", bbox_inches='tight')
plt.show()

# ------------------------------------------------------
# Step 7: Insights Summary
# ------------------------------------------------------
summary = """
📌 Titanic Dataset Insights:

1️⃣ Females had a significantly higher survival rate than males.
2️⃣ Passengers in 1st class had better survival chances compared to 2nd and 3rd class.
3️⃣ Children (under 12) had a higher survival rate compared to adults.
4️⃣ Higher fare generally indicated better survival probability.
5️⃣ Missing values were mainly in 'age', 'deck', and 'embark_town' — handled appropriately.
"""

print(summary)

# ------------------------------------------------------
# Step 8: Save Outputs
# ------------------------------------------------------
df.to_csv("titanic_cleaned.csv", index=False)
with open("titanic_insights.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n✅ All graphs displayed and saved in 'graphs/' folder.")
print("✅ Cleaned dataset and insights file exported successfully!")
