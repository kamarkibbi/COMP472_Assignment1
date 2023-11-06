import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the abalone dataset
abalone_df = pd.read_csv('C:\\Users\\moniq\\Desktop\\Comp 472\\Assignment1\\A1\\COMP472-A1-datasets\\abalone.csv')

# Load the penguins dataset
penguins_df = pd.read_csv('C:\\Users\\moniq\\Desktop\\Comp 472\\Assignment1\\A1\\COMP472-A1-datasets\\penguins.csv')

# Convert "island" and "sex" into 1-hot encoded vectors
penguins_df = pd.get_dummies(penguins_df, columns=["island", "sex"])

# Display information about the penguins dataset
print("Penguins Dataset:")
print(penguins_df.info())
print(penguins_df.columns)

# Convert the "Type" column to numerical categories in the abalone dataset
label_encoder = LabelEncoder()
abalone_df["Type"] = label_encoder.fit_transform(abalone_df["Type"])

# Display information about the abalone dataset
print("\nAbalone Dataset:")
print(abalone_df.info())


