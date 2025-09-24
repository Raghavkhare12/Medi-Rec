import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/Training.csv")

symptom_cols = df.columns.drop('prognosis')


symptom_counts = (df[symptom_cols] != 0).sum().sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.plot(symptom_counts.values, symptom_counts.index, 'o', markersize=7)

plt.xlabel("Frequency")
plt.ylabel("Symptoms")
plt.title("Dot Plot of Symptom Frequencies in Training Data")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
