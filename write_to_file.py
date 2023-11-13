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


def write_to_file(fileName, y_test, y_pred_base_dt, y_pred_top_dt, y_pred_base_mlp, y_pred_top_mlp,
                  dt_best_params, mlp_best_params):
    with open(fileName, 'a') as file:

        '''part b'''
        confusion_matrix_base_dt = metrics.confusion_matrix(y_test, y_pred_base_dt)
        '''part c'''
        precision_base_dt_macro = metrics.precision_score(y_test, y_pred_base_dt, average="macro")
        precision_base_dt_micro = metrics.precision_score(y_test, y_pred_base_dt, average="micro")
        precision_base_dt_weighted = metrics.precision_score(y_test, y_pred_base_dt, average="weighted")
        recall_base_dt_macro = metrics.recall_score(y_test, y_pred_base_dt, average="macro")
        recall_base_dt_micro = metrics.recall_score(y_test, y_pred_base_dt, average="micro")
        recall_base_dt_weighted = metrics.recall_score(y_test, y_pred_base_dt, average="weighted")
        f1_score_base_dt_macro = metrics.f1_score(y_test, y_pred_base_dt, average="macro")
        f1_score_base_dt_micro = metrics.f1_score(y_test, y_pred_base_dt, average="micro")
        f1_score_base_dt_weighted = metrics.f1_score(y_test, y_pred_base_dt, average="weighted")
        '''part d'''
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
            "Precision Micro: " + str(precision_base_dt_micro) + "\n" +
            "Precision Macro: " + str(precision_base_dt_macro) + "\n" +
            "Precision Weighted: " + str(precision_base_dt_weighted) + "\n" +
            "Recall Micro: " + str(recall_base_dt_micro) + "\n" +
            "Recall Macro: " + str(recall_base_dt_macro) + "\n" +
            "Recall Weighted: " + str(recall_base_dt_weighted) + "\n" +
            "F1-measure Micro: " + str(f1_score_base_dt_micro) + "\n" +
            "F1-measure Macro: " + str(f1_score_base_dt_macro) + "\n" +
            "F1-measure Weighted: " + str(f1_score_base_dt_weighted) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_base_dt) + "\n" +
            "Macro-Average F1: " + str(macro_f1_base_dt) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_base_dt) + "\n\n"
        )

        '''part b'''
        confusion_matrix_top_dt = metrics.confusion_matrix(y_test, y_pred_top_dt)
        '''part c'''
        precision_top_dt_macro = metrics.precision_score(y_test, y_pred_top_dt, average="macro")
        precision_top_dt_micro = metrics.precision_score(y_test, y_pred_top_dt, average="micro")
        precision_top_dt_weighted = metrics.precision_score(y_test, y_pred_top_dt, average="weighted")
        recall_top_dt_macro = metrics.recall_score(y_test, y_pred_top_dt, average="macro")
        recall_top_dt_micro = metrics.recall_score(y_test, y_pred_top_dt, average="micro")
        recall_top_dt_weighted = metrics.recall_score(y_test, y_pred_top_dt, average="weighted")
        f1_score_top_dt_macro = metrics.f1_score(y_test, y_pred_top_dt, average="macro")
        f1_score_top_dt_micro = metrics.f1_score(y_test, y_pred_top_dt, average="micro")
        f1_score_top_dt_weighted = metrics.f1_score(y_test, y_pred_top_dt, average="weighted")
        '''part d'''
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
            "Precision Micro: " + str(precision_top_dt_micro) + "\n" +
            "Precision Macro: " + str(precision_top_dt_macro) + "\n" +
            "Precision Weighted: " + str(precision_top_dt_weighted) + "\n" +
            "Recall Micro: " + str(recall_top_dt_micro) + "\n" +
            "Recall Macro: " + str(recall_top_dt_macro) + "\n" +
            "Recall Weighted: " + str(recall_top_dt_weighted) + "\n" +
            "F1-measure Micro: " + str(f1_score_top_dt_micro) + "\n" +
            "F1-measure Macro: " + str(f1_score_top_dt_macro) + "\n" +
            "F1-measure Weighted: " + str(f1_score_top_dt_weighted) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_top_dt) + "\n" +
            "Macro-Average F1: " + str(macro_f1_top_dt) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_top_dt) + "\n\n"
        )

        '''part b'''
        confusion_matrix_base_mlp = metrics.confusion_matrix(y_test, y_pred_base_mlp)
        '''part c'''
        precision_base_mlp_macro = metrics.precision_score(y_test, y_pred_base_mlp, average="macro")
        precision_base_mlp_micro = metrics.precision_score(y_test, y_pred_base_mlp, average="micro")
        precision_base_mlp_weighted = metrics.precision_score(y_test, y_pred_base_mlp, average="weighted")
        recall_base_mlp_macro = metrics.recall_score(y_test, y_pred_base_mlp, average="macro")
        recall_base_mlp_micro = metrics.recall_score(y_test, y_pred_base_mlp, average="micro")
        recall_base_mlp_weighted = metrics.recall_score(y_test, y_pred_base_mlp, average="weighted")
        f1_score_base_mlp_macro = metrics.f1_score(y_test, y_pred_base_mlp, average="macro")
        f1_score_base_mlp_micro = metrics.f1_score(y_test, y_pred_base_mlp, average="micro")
        f1_score_base_mlp_weighted = metrics.f1_score(y_test, y_pred_base_mlp, average="weighted")
        '''part d'''
        accuracy_base_mlp = metrics.accuracy_score(y_test, y_pred_base_mlp)
        macro_f1_base_mlp = metrics.f1_score(y_test, y_pred_base_mlp, average="macro")
        weighted_f1_base_mlp = metrics.f1_score(y_test, y_pred_base_mlp, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Base-MLP \n" +
            "Hyper-parameters changed: None \n\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_base_mlp) + "\n\n" +
            "(C) \n" +
            "Precision Micro: " + str(precision_base_mlp_micro) + "\n" +
            "Precision Macro: " + str(precision_base_mlp_macro) + "\n" +
            "Precision Weighted: " + str(precision_base_mlp_weighted) + "\n" +
            "Recall Micro: " + str(recall_base_mlp_micro) + "\n" +
            "Recall Macro: " + str(recall_base_mlp_macro) + "\n" +
            "Recall Weighted: " + str(recall_base_mlp_weighted) + "\n" +
            "F1-measure Micro: " + str(f1_score_base_mlp_micro) + "\n" +
            "F1-measure Macro: " + str(f1_score_base_mlp_macro) + "\n" +
            "F1-measure Weighted: " + str(f1_score_base_mlp_weighted) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_base_mlp) + "\n" +
            "Macro-Average F1: " + str(macro_f1_base_mlp) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_base_mlp) + "\n\n"
        )

        '''part b'''
        confusion_matrix_top_mlp = metrics.confusion_matrix(y_test, y_pred_top_mlp)
        '''part c'''
        precision_top_mlp_macro = metrics.precision_score(y_test, y_pred_top_mlp, average="macro")
        precision_top_mlp_micro = metrics.precision_score(y_test, y_pred_top_mlp, average="micro")
        precision_top_mlp_weighted = metrics.precision_score(y_test, y_pred_top_mlp, average="weighted")
        recall_top_mlp_macro = metrics.recall_score(y_test, y_pred_top_mlp, average="macro")
        recall_top_mlp_micro = metrics.recall_score(y_test, y_pred_top_mlp, average="micro")
        recall_top_mlp_weighted = metrics.recall_score(y_test, y_pred_top_mlp, average="weighted")
        f1_score_top_mlp_macro = metrics.f1_score(y_test, y_pred_top_mlp, average="macro")
        f1_score_top_mlp_micro = metrics.f1_score(y_test, y_pred_top_mlp, average="micro")
        f1_score_top_mlp_weighted = metrics.f1_score(y_test, y_pred_top_mlp, average="weighted")
        '''part d'''
        accuracy_top_mlp = metrics.accuracy_score(y_test, y_pred_top_mlp)
        macro_f1_top_mlp = metrics.f1_score(y_test, y_pred_top_mlp, average="macro")
        weighted_f1_top_mlp = metrics.f1_score(y_test, y_pred_top_mlp, average="weighted")

        file.write(
            "*******************************************************************************\n\n" +
            "(A)\n" +
            "Model: Top-MLP using gridsearch \n" +
            "Hyper-parameters changed: \n" + str(mlp_best_params) + "\n" +
            "(B) \n" +
            "Confusion Matrix: \n" + str(confusion_matrix_top_mlp) + "\n\n" +
            "(C) \n" +
            "Precision Micro: " + str(precision_top_mlp_micro) + "\n" +
            "Precision Macro: " + str(precision_top_mlp_macro) + "\n" +
            "Precision Weighted: " + str(precision_top_mlp_weighted) + "\n" +
            "Recall Micro: " + str(recall_top_mlp_micro) + "\n" +
            "Recall Macro: " + str(recall_top_mlp_macro) + "\n" +
            "Recall Weighted: " + str(recall_top_mlp_weighted) + "\n" +
            "F1-measure Micro: " + str(f1_score_top_mlp_micro) + "\n" +
            "F1-measure Macro: " + str(f1_score_top_mlp_macro) + "\n" +
            "F1-measure Weighted: " + str(f1_score_top_mlp_weighted) + "\n" +
            "(D) \n" +
            "Accuracy: " + str(accuracy_top_mlp) + "\n" +
            "Macro-Average F1: " + str(macro_f1_top_mlp) + "\n" +
            "Weighted-Average F1: " + str(weighted_f1_top_mlp) + "\n\n"
        )

        
