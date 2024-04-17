import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('Heart_Attack.csv')

# Function to remove outliers using the IQR method
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df >= lower_bound) & (df <= upper_bound)].dropna()

# Apply the outlier removal function to all columns except the outcome column
cleaned_data = remove_outliers(data.drop(columns=['heart_attack']))

# Add the outcome column back
cleaned_data['heart_attack'] = data['heart_attack']

# Check for missing values and impute if necessary (using median for demonstration)
for column in cleaned_data.columns[:-1]:
    if cleaned_data[column].isna().sum() > 0: column
    cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
columns_to_normalize = cleaned_data.columns[:-1]  # Excluding the outcome column
cleaned_data[columns_to_normalize] = scaler.fit_transform(cleaned_data[columns_to_normalize])

# Save the processed data
cleaned_data.to_csv('Processed_Heart_Attack.csv', index=False)

print(cleaned_data.head())
