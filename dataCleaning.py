import pandas as pd
#load in the dataset to a dataframe
df = pd.read_csv('datasets/loan_data.csv')

#remove all duplicates
df = df.drop_duplicates()

#remove all rows with missing values
df = df.dropna()

# Perform one-hot encoding on specific columns
# This converts all 0/1 values and all 'yes'/'no' values to True or False.
# Takes every category in columns of the DataFrame and converts them into a new column with a True or False value.
# Gender is also changed from 'male' or 'female' to a new column 'person_gender_male' which holds a True or False value (False means female).
categorical_columns = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file', 'loan_status']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the cleaned and encoded DataFrame to a CSV file
df_encoded.to_csv('cleaned_loan_data.csv', index=False)
