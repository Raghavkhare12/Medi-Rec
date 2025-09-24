import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load or Create Model and Data ---

# Try to load your pre-trained model to get its parameters
try:
    with open('models/svc.pkl', 'rb') as f:
        svc_model = pickle.load(f)
    print("Loaded pre-trained SVC model.")
    # Use a dummy classifier for creating data with the same characteristics
    n_features = svc_model.n_features_in_
    n_classes = len(svc_model.classes_)
except FileNotFoundError:
    print("Note: 'models/svc.pkl' not found. Using a dummy SVC model for demonstration.")
    n_features = 20
    n_classes = 5

# For demonstration, we'll create a synthetic dataset.
# In your actual use case, you would load your full training dataset here.
# FIX: Increased n_informative from 5 to 7 to satisfy the condition
# that n_classes * n_clusters_per_class <= 2**n_informative
X, y = make_classification(
    n_samples=1000,
    n_features=n_features,
    n_informative=7, # Changed from 5 to 7
    n_redundant=0,
    n_classes=n_classes,
    random_state=42
)

# --- 2. Generate Validation Curve ---

# Define the hyperparameter range to test. We'll check 'gamma'.
# Gamma defines how much influence a single training example has.
param_range = np.logspace(-6, -1, 10) # 10 values from 10^-6 to 10^-1

# Calculate training and validation scores using the validation_curve function.
# This function performs cross-validation for each hyperparameter value.
train_scores, test_scores = validation_curve(
    SVC(),              # The model to evaluate
    X,                  # The feature data
    y,                  # The target labels
    param_name="gamma", # The hyperparameter to vary
    param_range=param_range,
    scoring="accuracy", # The performance metric
    n_jobs=-1,          # Use all available CPU cores
    cv=3                # Number of cross-validation folds
)

# Calculate the mean and standard deviation for train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# --- 3. Plot the Validation Curve ---

plt.figure(figsize=(10, 6))
plt.title("Validation Curve for SVC (gamma)", fontsize=16, fontweight='bold')
plt.xlabel("γ (gamma)", fontsize=12)
plt.ylabel("Score (Accuracy)", fontsize=12)
plt.ylim(0.0, 1.1)
lw = 2

# Plot the training score curve
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
# Add a shaded area to show the variance (standard deviation)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange")

# Plot the cross-validation (test) score curve
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
# Add a shaded area for variance
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy")

plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n--- Validation Curve Summary ---")
print("This plot shows how the SVC's accuracy changes as the 'gamma' parameter is varied.")
print("- A large gap between the training score and cross-validation score suggests high variance (overfitting).")
print("- If both scores are low, it suggests high bias (underfitting).")
print("The optimal 'gamma' is typically where the cross-validation score is maximized.")

