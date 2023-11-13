
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from write_to_file import write_to_file

# Load the penguins dataset
penguins_df = pd.read_csv('penguins.csv')

# Display the first few rows of the DataFrames
print(penguins_df.head())

'''Step 1''' 

#Manually categorize island and sex of the penguins dataset using label encoder
label_encoder = LabelEncoder()
penguins_df[["island", "sex"]] = penguins_df[["island", "sex"]].apply(label_encoder.fit_transform)


# Display information about the penguins label encoded dataset
print("\nPenguins label-encoded Dataset information:")
print(penguins_df.info())


'''Step 2'''
#
# # Get the percentage of instances in each output class - penguins:
# plt.style.use('bmh')
#
# column_names = ['species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
#
# for col_name in column_names:
#
#     value_counts = penguins_df[col_name].value_counts()
#     percentages = (value_counts / value_counts.sum()) * 100
#
#     if len(percentages) > 200:
#         plt.figure(figsize=(40, 6))
#     elif len(percentages) > 50:
#         plt.figure(figsize=(25, 6))
#     else:
#         plt.figure(figsize=(15, 6))
#
#     percentages.plot(kind='bar', color='skyblue')
#     plt.title('Percentages of ' + col_name)
#     plt.xlabel(col_name)
#     plt.ylabel('Percentage')
#
#     plt.savefig('penguin-classes/' + col_name + '.png', bbox_inches='tight')
#
# # Get the percentage of instances in each output class - abalone:
# plt.style.use('bmh')
#
# column_names = ['Type', 'LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
#
# for col_name in column_names:
#     value_counts = abalone_df[col_name].value_counts()
#     percentages = (value_counts / value_counts.sum()) * 100
#
#     if len(percentages) > 500:
#         plt.figure(figsize=(200, 6))
#     elif len(percentages) > 200:
#         plt.figure(figsize=(40, 6))
#     elif len(percentages) > 50:
#         plt.figure(figsize=(25, 6))
#     else:
#         plt.figure(figsize=(15, 6))
#
#     percentages.plot(kind='bar', color='skyblue')
#     plt.title('Percentages of ' + col_name)
#     plt.xlabel(col_name)
#     plt.ylabel('Percentage')
#
#     plt.savefig('abalone-classes/' + col_name + '.png', bbox_inches='tight')

'''Step 3'''
#Use train-test-split to split all three datasets using default parameter values

#Split the label-encoded (manually-categorized) peguins dataset
df_X=penguins_df[['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']]
df_y=penguins_df[['species']]


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)

'''Step 4'''

'''Step 4a'''
# Create and fit a Decision Tree classifier with default parameters
base_dt = DecisionTreeClassifier()

# for the label-encoded penguins dataset
base_dt.fit(X_train, y_train)
y_pred_base_dt=base_dt.predict(X_test)

'''Step 4b'''

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10], 
    'min_samples_split': [2, 5, 10]  
}


# Create a Decision Tree classifier
top_dt = DecisionTreeClassifier()

# Perform grid search
grid_search = GridSearchCV(top_dt, param_grid, cv=5)

# for the label-encoded penguins dataset
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params_dt = grid_search.best_params_
best_estimator_dt = grid_search.best_estimator_

# for the label-encoded penguins dataset
y_pred_top_dt=best_estimator_dt.predict(X_test)


def visualize_decision_tree(dt_model, X_train, y_train):
    plt.figure(figsize=(12, 8))
    plot_tree(
        dt_model,
        feature_names=X_train.columns.tolist(),
        class_names=y_train['species'].unique().tolist(),
        filled=True
    )
    plt.show()

# Visualize the base Decision Tree
visualize_decision_tree(base_dt, X_train, y_train)

# Visualize the best Decision Tree found using GridSearchCV
visualize_decision_tree(best_estimator_dt, X_train, y_train)

'''Step 4c'''
# Create and fit a Multi-Layer Perceptron (MLP) classifier with default parameters
base_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),  
    activation='logistic',
    solver='sgd', 
)

# Convert the target y column of values to an array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train the model
base_mlp.fit(X_train, y_train)
# Test the model
y_pred_base_mlp = base_mlp.predict(X_test)

'''Step 4d'''
# Define hyperparameter grid for GridSearchCV
mlp_param_grid ={
    'hidden_layer_sizes':[(30, 50), (10, 10, 10)], 
    'activation':['logistic', 'tanh', 'relu'], #Logistic = Sigmoid
    'solver': ['adam', 'sgd']
}

# Create an MLP classifier
top_mlp = MLPClassifier()

# Perform grid search
mlp_grid_search = GridSearchCV(top_mlp, mlp_param_grid, cv=5)
mlp_grid_search.fit(X_train, y_train)

# Get the best parameters
mlp_best_params = mlp_grid_search.best_params_
mlp_best_estimator = mlp_grid_search.best_estimator_

# Test the labeled-encoded penguins dataset
y_pred_top_mlp = mlp_best_estimator.predict(X_test)


'''Step 5'''
with open('penguins_le.txt', 'w') as f:
    f.write("Single Run Output for part 5: \n\n")
write_to_file('penguins_le.txt', y_test, y_pred_base_dt, y_pred_top_dt, y_pred_base_mlp, y_pred_top_mlp,
              best_params_dt, mlp_best_params)

'''Step 6'''
#Redo steps 4 & 5, 5 times for each model and append in the performance files
with open('penguins_le.txt', 'a') as f:
    f.write("Step 6 Output: \n\n")
num_iterations = 5
accuracy_results = {'base_dt_Acc': [], 'top_dt_Acc': [], 'base_mlp_Acc': [], 'top_mlp_Acc': []}
macro_f1_results = {'base_dt_Macf1': [], 'top_dt_Macf1': [], 'base_mlp_Macf1': [], 'top_mlp_Macf1': []}
weighted_avg_f1_results = {'base_dt_Wavgf1': [], 'top_dt_Wavgf1': [], 'base_mlp_Wavgf1': [], 'top_mlp_Wavgf1': []}

for i in range(num_iterations):
    print(f"\nIteration {i + 1}/{num_iterations}")

    with open('penguins_le.txt', 'a') as f:
        f.write(f"\nIteration {i + 1}/{num_iterations}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)

    # Base Decision Tree
    base_dt.fit(X_train, y_train)
    y_pred_base_dt = base_dt.predict(X_test)
    accuracy_results['base_dt_Acc'].append(accuracy_score(y_test, y_pred_base_dt))
    macro_f1_results['base_dt_Macf1'].append(metrics.f1_score(y_test,y_pred_base_dt,average="macro"))
    weighted_avg_f1_results['base_dt_Wavgf1'].append(metrics.f1_score(y_test, y_pred_base_dt, average="weighted"))

    # Top Decision Tree
    grid_search.fit(X_train, y_train)
    best_estimator_dt = grid_search.best_estimator_
    y_pred_top_dt = best_estimator_dt.predict(X_test)
    accuracy_results['top_dt_Acc'].append(accuracy_score(y_test, y_pred_top_dt))
    macro_f1_results['top_dt_Macf1'].append(metrics.f1_score(y_test,y_pred_top_dt,average="macro"))
    weighted_avg_f1_results['top_dt_Wavgf1'].append(metrics.f1_score(y_test, y_pred_top_dt, average="weighted"))

    # Base MLP
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    base_mlp.fit(X_train, y_train)
    y_pred_base_mlp = base_mlp.predict(X_test)
    accuracy_results['base_mlp_Acc'].append(accuracy_score(y_test, y_pred_base_mlp))
    macro_f1_results['base_mlp_Macf1'].append(metrics.f1_score(y_test,y_pred_base_mlp,average="macro"))
    weighted_avg_f1_results['base_mlp_Wavgf1'].append(metrics.f1_score(y_test, y_pred_base_mlp, average="weighted"))

    # Top MLP
    mlp_grid_search.fit(X_train, y_train)
    mlp_best_estimator = mlp_grid_search.best_estimator_
    y_pred_top_mlp = mlp_best_estimator.predict(X_test)
    accuracy_results['top_mlp_Acc'].append(accuracy_score(y_test, y_pred_top_mlp))
    macro_f1_results['top_mlp_Macf1'].append(metrics.f1_score(y_test,y_pred_top_mlp,average="macro"))
    weighted_avg_f1_results['top_mlp_Wavgf1'].append(metrics.f1_score(y_test, y_pred_top_mlp, average="weighted"))

    write_to_file('penguins_le.txt', y_test, y_pred_base_dt, y_pred_top_dt, y_pred_base_mlp, y_pred_top_mlp, 
                  best_params_dt, mlp_best_params)

# Calculate average and variance for each model
for model in ['base_dt_Acc', 'top_dt_Acc', 'base_mlp_Acc', 'top_mlp_Acc']:
    avg_accuracy = np.mean(accuracy_results[model])
    var_accuracy = np.var(accuracy_results[model])

    with open('penguins_le.txt', 'a') as f:
        if(model=='base_dt_Acc'):
            f.write(f"\n\nModel: Base-DT\n")
        elif (model=='top_dt_Acc'):
            f.write(f"\n\nModel: Top-DT\n")
        elif (model=='base_mlp_Acc'):
            f.write(f"\n\nModel: Base-MLP\n")
        else:
            f.write(f"\n\nModel: Top-MLP\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}, Variance: {var_accuracy:.4f}\n")

for model in ['base_dt_Macf1', 'top_dt_Macf1', 'base_mlp_Macf1', 'top_mlp_Macf1']:
    avg_Macf1 = np.mean(macro_f1_results[model])
    var_Macf1 = np.var(macro_f1_results[model])

    with open('penguins_le.txt', 'a') as f:
        if(model=='base_dt_Macf1'):
            f.write(f"\n\nModel: Base-DT\n")
        elif (model=='top_dt_Macf1'):
            f.write(f"\n\nModel: Top-DT\n")
        elif (model=='base_mlp_Macf1'):
            f.write(f"\n\nModel: Base-MLP\n")
        else:
            f.write(f"\n\nModel: Top-MLP\n")
        f.write(f"Average Macro Average F1: {avg_Macf1:.4f}, Variance: {var_Macf1:.4f}\n")

for model in ['base_dt_Wavgf1', 'top_dt_Wavgf1', 'base_mlp_Wavgf1', 'top_mlp_Wavgf1']:
    avg_Wavgf1 = np.mean(weighted_avg_f1_results[model])
    var_Wavgf1 = np.var(weighted_avg_f1_results[model])

    with open('penguins_le.txt', 'a') as f:
        if(model=='base_dt_Wavgf1'):
            f.write(f"\n\nModel: Base-DT\n")
        elif (model=='top_dt_Wavgf1'):
            f.write(f"\n\nModel: Top-DT\n")
        elif (model=='base_mlp_Wavgf1'):
            f.write(f"\n\nModel: Base-MLP\n")
        else:
            f.write(f"\n\nModel: Top-MLP\n")
        f.write(f"Average Weighted Average F1: {avg_Wavgf1:.4f}, Variance: {var_Wavgf1:.4f}\n")
