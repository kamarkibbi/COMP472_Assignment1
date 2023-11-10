from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Load the abalone dataset
abalone_df = pd.read_csv('abalone.csv')


# Display the first few rows of the DataFrames
print(abalone_df.head())

'''Step 1''' 

#no need to encode anything from the abalone dataset

# Display information about the abalone dataset
print("\nAbalone Dataset:")
print(abalone_df.info())

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

#Split the abalone dataset
abalone_df_X=abalone_df[['LongestShell','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']]
abalone_df_y=abalone_df[['Type']]

abalone_df_X_train, abalone_df_X_test, abalone_df_y_train, abalone_df_y_test=train_test_split(abalone_df_X,abalone_df_y)

'''Step 4'''

'''Step 4a'''
# Create and fit a Decision Tree classifier with default parameters
base_dt = DecisionTreeClassifier()

# for the label-encoded penguins dataset
base_dt.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)
penguins_LE_df_y_pred=base_dt.predict(penguins_LE_df_X_test)

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
grid_search.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# for the label-encoded penguins dataset
penguins_LE_df_y_pred=best_estimator.predict(penguins_LE_df_X_test)

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
visualize_decision_tree(base_dt, penguins_LE_df_X_train, penguins_LE_df_y_train)

# Visualize the best Decision Tree found using GridSearchCV
visualize_decision_tree(best_estimator, penguins_LE_df_X_train, penguins_LE_df_y_train)



'''
# Create and fit a Multi-Layer Perceptron (MLP) classifier with default parameters
base_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),  
    activation='logistic',
    solver='sgd',
    max_iter=2000,  
    learning_rate='constant',  
    learning_rate_init=0.01  
)
base_mlp.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)


# Define hyperparameter grid for GridSearchCV
mlp_param_grid = {
    'activation': ['sigmoid', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  
    'solver': ['adam', 'sgd']
}

# Create an MLP classifier
top_mlp = MLPClassifier()

# Perform grid search
mlp_grid_search = GridSearchCV(top_mlp, mlp_param_grid, cv=5)
mlp_grid_search.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)

# Get the best parameters
mlp_best_params = mlp_grid_search.best_params_

# Create and fit an MLP classifier with the best parameters
best_mlp = MLPClassifier(**mlp_best_params)
best_mlp.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)

'''

