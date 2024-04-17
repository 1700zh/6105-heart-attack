import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the processed data
data = pd.read_csv('Processed_Heart_Attack.csv')

# Splitting the data into features and target
X = data.drop(columns=['heart_attack'])
y = data['heart_attack']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert the PCA results into a DataFrame for easier handling and saving
train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
test_pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'])

# Add the target column back to the PCA-transformed data
train_pca_df['heart_attack'] = y_train.reset_index(drop=True)
test_pca_df['heart_attack'] = y_test.reset_index(drop=True)

# Specify the correct path where you want to save the files
train_pca_df.to_csv('PCA_Training_Data.csv', index=False)
test_pca_df.to_csv('PCA_Test_Data.csv', index=False)

print("Training and test data with PCA applied have been saved.")
