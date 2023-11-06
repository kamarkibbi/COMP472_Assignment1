import pandas as pd


# Load the abalone dataset
abalone_df = pd.read_csv('C:\\Users\\moniq\\Desktop\\Comp 472\\Assignment1\\A1\\COMP472-A1-datasets\\abalone.csv')

# Load the penguins dataset
penguins_df = pd.read_csv('C:\\Users\\moniq\\Desktop\\Comp 472\\Assignment1\\A1\\COMP472-A1-datasets\\penguins.csv')


# Display the first few rows of the DataFrame
print(abalone_df.head())
print(penguins_df.head())


