from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the abalone dataset
abalone_df = pd.read_csv('C:\\Users\\kamar\\OneDrive\\Desktop\\College shiz\\Concordia\\Fall 2023\\COMP 472\\Assignment 1\\COMP472_Assignment1\\abalone.csv')

# Load the penguins dataset
penguins_df = pd.read_csv('C:\\Users\\kamar\\OneDrive\\Desktop\\College shiz\\Concordia\\Fall 2023\\COMP 472\\Assignment 1\\COMP472_Assignment1\\penguins.csv')

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
print(penguins_df.info())





