import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pickle


TRAINING_DATA_PATH = 'datasets/Training.csv'
MODEL_PATH = 'models/svc.pkl'

try:
   
    df = pd.read_csv(TRAINING_DATA_PATH)

    
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']

   
    le = LabelEncoder()
    Y = le.fit_transform(y)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    
    with open(MODEL_PATH, 'rb') as f:
        svc = pickle.load(f)

   
    predictions = svc.predict(X_test)

   
    cm = confusion_matrix(y_test, predictions)

    
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='viridis')
    plt.title('Confusion Matrix Heatmap', fontsize=20)
    plt.xlabel('Predicted Disease', fontsize=15)
    plt.ylabel('Actual Disease', fontsize=15)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

   
    plt.savefig('confusion_matrix_heatmap.png')
    print("Confusion matrix heatmap has been generated and saved as 'confusion_matrix_heatmap.png'")


except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nPlease make sure that 'Training.csv' is inside a 'datasets' folder,")
    print("and 'svc.pkl' is in the same directory as your script.")
    print("If your file structure is different, please update the 'TRAINING_DATA_PATH' and 'MODEL_PATH' variables in the code.")