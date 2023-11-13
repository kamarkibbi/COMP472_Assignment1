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
from write_to_file import write_to_file
from sklearn.metrics import accuracy_score
from computations import computations
import numpy as np

'''Step 1''' 

# Load the penguins dataset
penguins_df = pd.read_csv('penguins.csv')

# Display the first few rows of the DataFrames
print(penguins_df.head())

#hot-encoded island and sex of the penguins dataset
cols = ['island', 'sex']
penguins_HotEncoded_df=pd.get_dummies(penguins_df,columns=cols)
print(penguins_HotEncoded_df.head())


# Display information about the penguins hot-encoded dataset
print("\nPenguins hot-encoded Dataset information:")
print(penguins_HotEncoded_df.info())

#Manually categorize island and sex of the penguins dataset using label encoder
label_encoder = LabelEncoder()
penguins_df[["island", "sex"]] = penguins_df[["island", "sex"]].apply(label_encoder.fit_transform)


# Display information about the penguins label encoded dataset
print("\nPenguins label-encoded Dataset information:")
print(penguins_df.info())

# Load the abalone dataset
abalone_df = pd.read_csv('abalone.csv')

# Display the first few rows of the DataFrames
print(abalone_df.head())

# no need to encode anything from the abalone dataset

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
# Use train-test-split to split all three datasets using default parameter values

df_X=penguins_HotEncoded_df[['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','island_Biscoe','island_Dream','island_Torgersen','sex_FEMALE','sex_MALE']]
df_y=penguins_HotEncoded_df[['species']]

computations('penguins_he.txt',df_X,df_y,'species')

df_X=penguins_df[['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']]
df_y=penguins_df[['species']]

computations('penguins_le.txt',df_X,df_y,'species')

df_X = abalone_df[
    ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']]
df_y = abalone_df[['Type']]

computations('abalone.txt',df_X,df_y,'Type')

