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

# Load the penguins dataset
penguins_df = pd.read_csv('penguins.csv')

# Display the first few rows of the DataFrames
print(abalone_df.head())
print(penguins_df.head())

'''Step 1''' 
#hot-encoded island and sex of the penguins dataset
cols = ['island', 'sex']
penguins_HotEncoded_df=pd.get_dummies(penguins_df,columns=cols)
print(penguins_HotEncoded_df.head())


#Manually categorize island and sex of the penguins dataset using label encoder
label_encoder = LabelEncoder()
penguins_df[["island", "sex"]] = penguins_df[["island", "sex"]].apply(label_encoder.fit_transform)

#no need to encode anything from the abalone dataset


# Display information about the penguins hot-encoded dataset
print("\nPenguins hot-encoded Dataset information:")
print(penguins_HotEncoded_df.info())

# Display information about the penguins label encoded dataset
print("\nPenguins label-encoded Dataset information:")
print(penguins_df.info())

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
#Split the hot-encoded penguins dataset
penguins_HE_df_X=penguins_HotEncoded_df[['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','island_Biscoe','island_Dream','island_Torgersen','sex_FEMALE','sex_MALE']]
penguins_HE_df_y=penguins_HotEncoded_df[['species']]

penguins_HE_df_X_train, penguins_HE_df_X_test, penguins_HE_df_y_train, penguins_HE_df_y_test=train_test_split(penguins_HE_df_X,penguins_HE_df_y)

#Split the label-encoded (manually-categorized) peguins dataset
penguins_LE_df_X=penguins_df[['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']]
penguins_LE_df_y=penguins_df[['species']]

penguins_LE_df_X_train, penguins_LE_df_X_test, penguins_LE_df_y_train, penguins_LE_df_y_test=train_test_split(penguins_LE_df_X,penguins_LE_df_y)

#Split the abalone dataset
abalone_df_X=abalone_df[['LongestShell','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']]
abalone_df_y=abalone_df[['Type']]

abalone_df_X_train, abalone_df_X_test, abalone_df_y_train, abalone_df_y_test=train_test_split(abalone_df_X,abalone_df_y)

'''Step 4'''
# for the label-encoded peguins dataset
# Create and fit a Decision Tree classifier with default parameters
base_dt = DecisionTreeClassifier()
base_dt.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)

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
grid_search.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Create and fit a Decision Tree classifier with the best parameters
best_dt = DecisionTreeClassifier(**best_params)
best_dt.fit(penguins_LE_df_X_train, penguins_LE_df_y_train)


def visualize_decision_tree(dt_model, X_train, y_train):
    plt.figure(figsize=(12, 8))
    plot_tree(dt_model, feature_names=X_train.columns, class_names=y_train['species'].unique(), filled=True)
    plt.show()

# Visualize the base Decision Tree
visualize_decision_tree(base_dt, penguins_LE_df_X_train, penguins_LE_df_y_train)

# Visualize the best Decision Tree found using GridSearchCV
visualize_decision_tree(best_dt, penguins_LE_df_X_train, penguins_LE_df_y_train)

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
    'activation': ['logistic', 'tanh', 'relu'],
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

