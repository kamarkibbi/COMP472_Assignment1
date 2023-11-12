import sklearn.metrics
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def write_to_file(y_test, y_pred_base_dt, y_pred_top_dt, y_pred_base_mlp, y_pred_top_mlp,
                  dt_best_params, mlp_best_params):
    with open('abalone.txt', 'w') as file:
        file.write("The below was generated using the Abalone Dataset\n\n")

        confusion_matrix_base_dt = metrics.confusion_matrix(y_test, y_pred_base_dt)
        precision_base_dt = metrics.precision_score(y_test, y_pred_base_dt, average="macro")
        recall_base_dt = metrics.recall_score(y_test, y_pred_base_dt, average="macro")
        f1_score_base_dt = metrics.f1_score(y_test, y_pred_base_dt, average="macro")
        accuracy_base_dt = metrics.accuracy_score(y_test, y_pred_base_dt)
        macro_f1_base_dt = metrics.f1_score(y_test, y_pred_base_dt, average="macro")
        weighted_f1_base_dt = metrics.f1_score(y_test, y_pred_base_dt, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Base-DT \n" +
            "Hyper-parameters changed: None \n\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_base_dt) + "\n\n" +
            "(C) \n" +
            "Precision: " + str(precision_base_dt) + "\n" +
            "Recall: " + str(recall_base_dt) + "\n" +
            "F1-measure: " + str(f1_score_base_dt) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_base_dt) + "\n" +
            "Macro-Average F1: " + str(macro_f1_base_dt) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_base_dt) + "\n\n"
        )

        confusion_matrix_top_dt = metrics.confusion_matrix(y_test, y_pred_top_dt)
        precision_top_dt = metrics.precision_score(y_test, y_pred_top_dt, average="macro")
        recall_top_dt = metrics.recall_score(y_test, y_pred_top_dt, average="macro")
        f1_score_top_dt = metrics.f1_score(y_test, y_pred_top_dt, average="macro")
        accuracy_top_dt = metrics.accuracy_score(y_test, y_pred_top_dt)
        macro_f1_top_dt = metrics.f1_score(y_test, y_pred_top_dt, average="macro")
        weighted_f1_top_dt = metrics.f1_score(y_test, y_pred_top_dt, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Top-DT using gridsearch \n" +
            "Hyper-parameters changed: \n" + str(dt_best_params) + "\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_top_dt) + "\n\n" +
            "(C) \n" +
            "Precision: " + str(precision_top_dt) + "\n" +
            "Recall: " + str(recall_top_dt) + "\n" +
            "F1-measure: " + str(f1_score_top_dt) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_top_dt) + "\n" +
            "Macro-Average F1: " + str(macro_f1_top_dt) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_top_dt) + "\n\n"
        )

        confusion_matrix_base_mlp = metrics.confusion_matrix(y_test, y_pred_base_mlp)
        precision_base_mlp = metrics.precision_score(y_test, y_pred_base_mlp, average="macro")
        recall_base_mlp = metrics.recall_score(y_test, y_pred_base_mlp, average="macro")
        f1_score_base_mlp = metrics.f1_score(y_test, y_pred_base_mlp, average="macro")
        accuracy_base_mlp = metrics.accuracy_score(y_test, y_pred_base_mlp)
        macro_f1_base_mlp = metrics.f1_score(y_test, y_pred_base_mlp, average="macro")
        weighted_f1_base_mlp = metrics.f1_score(y_test, y_pred_base_mlp, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Base-DT \n" +
            "Hyper-parameters changed: None \n\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_base_mlp) + "\n\n" +
            "(C) \n" +
            "Precision: " + str(precision_base_mlp) + "\n" +
            "Recall: " + str(recall_base_mlp) + "\n" +
            "F1-measure: " + str(f1_score_base_mlp) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_base_mlp) + "\n" +
            "Macro-Average F1: " + str(macro_f1_base_mlp) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_base_mlp) + "\n\n"
        )

        confusion_matrix_top_mlp = metrics.confusion_matrix(y_test, y_pred_top_mlp)
        precision_top_mlp = metrics.precision_score(y_test, y_pred_top_mlp, average="macro")
        recall_top_mlp = metrics.recall_score(y_test, y_pred_top_mlp, average="macro")
        f1_score_top_mlp = metrics.f1_score(y_test, y_pred_top_mlp, average="macro")
        accuracy_top_mlp = metrics.accuracy_score(y_test, y_pred_top_mlp)
        macro_f1_top_mlp = metrics.f1_score(y_test, y_pred_top_mlp, average="macro")
        weighted_f1_top_mlp = metrics.f1_score(y_test, y_pred_top_mlp, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Top-DT using gridsearch \n" +
            "Hyper-parameters changed: \n" + str(mlp_best_params) + "\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_top_mlp) + "\n\n" +
            "(C) \n" +
            "Precision: " + str(precision_top_mlp) + "\n" +
            "Recall: " + str(recall_top_mlp) + "\n" +
            "F1-measure: " + str(f1_score_top_mlp) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_top_mlp) + "\n" +
            "Macro-Average F1: " + str(macro_f1_top_mlp) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_top_mlp) + "\n\n"
        )
