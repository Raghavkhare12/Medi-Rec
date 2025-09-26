# MediRec: AI-Powered Medical Recommendation System

MediRec is an intelligent, web-based application designed to provide users with a preliminary analysis of their medical symptoms using a machine learning model.

## 🌟 Features

  * **Symptom Analysis**: Select your symptoms from a comprehensive list.
  * **AI-Powered Predictions**: Utilizes a Support Vector Classifier (SVC) model to predict potential medical conditions based on your symptoms.
  * **Detailed Information**: For each prediction, receive detailed information, including:
      * **Description of the illness**: Understand the predicted condition.
      * **Recommended Precautions**: Learn about the steps you can take to manage the condition.
      * **Common Medications**: Get a list of common medications.
      * **Dietary Recommendations**: Find out what to eat and what to avoid.
      * **Suggested Workouts**: Get recommendations for exercises to help you feel better.

## ⚙️ How It Works

1.  **Select Symptoms**: Choose your symptoms from a list of 132 different symptoms.
2.  **Get Prediction**: The system uses a pre-trained SVC model to predict the disease.
3.  **View Results**: The application displays the predicted disease along with a wealth of information to help you understand and manage your health.

## 💻 Technology Stack

  * **Frontend**: HTML, CSS, JavaScript, Bootstrap
  * **Backend**: Python, Flask
  * **Machine Learning**: Scikit-learn, Pandas, NumPy

## 🚀 Getting Started

### Prerequisites

  * Python 3.x
  * pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/raghavkhare12/medi-rec.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd medi-rec
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the Flask application:
    ```bash
    python main.py
    ```
2.  Open your web browser and go to `http://127.0.0.1:5000`

## 📊 Dataset

The model was trained on the "Training.csv" dataset, which contains 4,920 records and 133 columns. Of these columns, 132 represent different symptoms, and the final column, 'prognosis', indicates the diagnosed disease. The dataset covers 41 unique diseases.

## 🤖 Model

Several machine learning models were trained and evaluated to find the best-performing algorithm for this classification task. The models tested include:

  * Support Vector Classifier (SVC) with a linear kernel
  * Random Forest Classifier
  * Gradient Boosting Classifier
  * K-Nearest Neighbors (KNN) Classifier
  * Multinomial Naive Bayes

### Model Performance

The Support Vector Classifier (SVC) was selected for the final application due to its efficiency and robustness, achieving a **99.7% accuracy** on the test set.

| Model | Accuracy | F1 Score |
| :--- | :--- | :--- |
| **Support Vector Classifier (SVC)** | **98.70%** | **98.80%** |
| Random Forest Classifier | 98.39% | 98.39% |
| Gradient Boosting Classifier | 97.26% | 97.10% |
| K-Nearest Neighbors (KNN) | 97.88% | 97.83% |
| Multinomial Naive Bayes | 98.17% | 98.19% |

Here is the confusion matrix for the SVC model:

## ⚠️ Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.