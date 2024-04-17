import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Load the PCA-transformed data
train_data = pd.read_csv('PCA_Training_Data.csv')
test_data = pd.read_csv('PCA_Test_Data.csv')

# Prepare training and test sets
X_train = train_data.drop('heart_attack', axis=1)
y_train = train_data['heart_attack']
X_test = test_data.drop('heart_attack', axis=1)
y_test = test_data['heart_attack']

# Define the base classifiers with an updated MLPClassifier
base_classifiers = [
    ('nb', GaussianNB()),
    ('nn', MLPClassifier(solver='sgd', max_iter=3000, learning_rate='adaptive', learning_rate_init=0.01, tol=1e-4, n_iter_no_change=20, hidden_layer_sizes=(50,), random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Meta-learner
dtree = DecisionTreeClassifier()

# Stacking classifier using a decision tree as a meta-learner
stack = StackingClassifier(estimators=base_classifiers, final_estimator=dtree, cv=5)

# Define parameter grid for each classifier
param_grid = {
    'nb__var_smoothing': [1e-09, 1e-08, 1e-10],
    'nn__activation': ['relu', 'tanh'],
    'knn__n_neighbors': [3, 5, 7],
    'final_estimator__max_depth': [None, 10, 20, 30],
    'final_estimator__min_samples_split': [2, 5, 10]
}

# Grid search to find the best parameters and fit model
grid_search = GridSearchCV(estimator=stack, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate on test data
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: {:.2f}".format(test_accuracy))
