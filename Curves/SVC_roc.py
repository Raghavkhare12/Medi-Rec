import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')


with open('models/svc.pkl', 'rb') as f:
    svc_model = pickle.load(f)


np.random.seed(42)
n_samples = 200
n_features = len(svc_model.feature_names_in_)
n_classes = len(svc_model.classes_)


X_test = np.random.randn(n_samples, n_features)
y_test = np.random.randint(0, n_classes, n_samples)


if hasattr(svc_model, 'predict_proba'):
    y_score = svc_model.predict_proba(X_test)
else:
   
    y_score = svc_model.decision_function(X_test)


y_test_bin = label_binarize(y_test, classes=svc_model.classes_)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure(figsize=(10, 8))


colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass (SVC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("AUC Scores:")
for i in range(n_classes):
    print(f"Class {i}: {roc_auc[i]:.3f}")
print(f"Micro-average: {roc_auc['micro']:.3f}")


if n_classes == 2:
   
    RocCurveDisplay.from_estimator(svc_model, X_test, y_test)
    plt.title('ROC Curve - Binary Classification')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()