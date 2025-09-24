import pandas as pd
from imblearn.over_sampling import SMOTE


df = pd.read_csv('datasets/Training_old.csv')

X = df.drop('prognosis', axis=1)
y = df['prognosis']


print("Class distribution before SMOTE:")
print(y.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=4)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass distribution after SMOTE:")
print(y_resampled.value_counts())

df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='prognosis')], axis=1)

df_balanced.to_csv('training.csv', index=False)

print("\nSMOTE applied successfully. The balanced data is saved in 'training.csv'")