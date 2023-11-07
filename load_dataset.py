from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# Load the abalone dataset
abalone_df = pd.read_csv('C:\\Users\\kamar\\OneDrive\\Desktop\\College shiz\\Concordia\\Fall 2023\\COMP 472\\Assignment 1\\COMP472_Assignment1\\abalone.csv')

# Load the penguins dataset
penguins_df = pd.read_csv('C:\\Users\\kamar\\OneDrive\\Desktop\\College shiz\\Concordia\\Fall 2023\\COMP 472\\Assignment 1\\COMP472_Assignment1\\penguins.csv')


# Display the first few rows of the DataFrame
#print(abalone_df.head())
print(penguins_df.head())

#step 1: 
cols = ['island', 'sex']
penguins_encoded_df=pd.get_dummies(penguins_df,columns=cols)
print(penguins_encoded_df.head())




