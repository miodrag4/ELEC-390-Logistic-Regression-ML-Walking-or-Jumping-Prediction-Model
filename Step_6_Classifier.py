import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import joblib
# Get data from Step_5_Feature_extraction.py
from Step_5_Feature_extraction import train_data, test_data

# Read datasets and create separate dataframes
featureNumber = 10
def filter_data_by_feature(dataset, feature_idx):
    return dataset[dataset.iloc[:, featureNumber] == feature_idx]

# Separate dataframes into train and test
datasets_train = [filter_data_by_feature(train_data, i) for i in range(1, 5)]
datasets_test = [filter_data_by_feature(test_data, i) for i in range(1, 5)]

# Separate dataframes into train and test segemtns and labels
X_train = [dataset.iloc[:, 0:-2] for dataset in datasets_train]
X_test = [dataset.iloc[:, 0:-2] for dataset in datasets_test]

# Labels (same for all dataframes)
Y_train = datasets_train[-1].iloc[:, -1]
Y_test = datasets_test[-1].iloc[:, -1]

# Combine the train and test sets with all the features for both.
X_combinedTrain = np.hstack(X_train)
X_combinedTest = np.hstack(X_test)

# Create a logistic regression model and train it with the combined train data , normalize the data
l_reg = LogisticRegression(max_iter=100000)
clfCombined = make_pipeline(StandardScaler(), l_reg)
clfCombined.fit(X_combinedTrain, Y_train)

# Save the model to a file called classifier.joblib
joblib.dump(clfCombined, 'classifier.joblib')

# Calculate the accuracy and recall of the model
Y_predicted_combined = clfCombined.predict(X_combinedTest)
Y_clf_prob_combined = clfCombined.predict_proba(X_combinedTest)

# Classification output
accuracyCombined = accuracy_score(Y_test, Y_predicted_combined)
recallCombined = recall_score(Y_test, Y_predicted_combined)
print('Accuracy of the model is:', accuracyCombined)
print('Recall of the model is:', recallCombined)

# Confusion matrix
cm = confusion_matrix(Y_test, Y_predicted_combined)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Receiver operating characteristic plot
fpr, tpr, _ = roc_curve(Y_test, Y_clf_prob_combined[:, 1], pos_label=clfCombined.classes_[1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title('ROC Curve')

# Set the limits of the y-axis
plt.ylim([0, 1.05])

plt.show()

# Calculate the overall F1 score
TP = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
F1 = (2 * TP) / (2 * TP + FP + FN)
print('F1 Score:', F1)

# Output area under the ROC curve
auc = roc_auc_score(Y_test, Y_clf_prob_combined[:, 1])
print('AUC:', auc)