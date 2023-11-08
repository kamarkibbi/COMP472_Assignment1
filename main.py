from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

