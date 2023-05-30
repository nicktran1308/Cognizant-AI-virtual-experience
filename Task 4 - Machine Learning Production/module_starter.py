

# --- 1) IMPORTING PACKAGES
import pandas as pd
"""
Random Forest Regressor:
    Belongs to the ensemble family.
    Specifically used for regression tasks. 
    It combines multiple decision trees to create a robust regression model.
"""
from sklearn.ensemble import RandomForestRegressor
"""
train_test_split: 
    Used to split the data into training and testing sets.
    It randomly shuffles and divides the data, evaluate the model's performance on unseen data.
"""
from sklearn.model_selection import train_test_split
"""
mean_absolute_error:
    Measure the Avg abs difference between predicted and actual values in a regression task. 
    It quantifies the model's accuracy by calculating the average absolute deviation of predictions from the ground truth.
"""
from sklearn.metrics import mean_absolute_error
"""
Standard Scaler:
    Data preprocessing technique that standardizes features by subtracting the mean and dividing by the standard deviation.
"""
from sklearn.preprocessing import StandardScaler

# --- 2) DEFINE GLOBAL CONSTANTS
# K is used to define the number of folds that will be used for cross-validation
K = 10 # Assess the performance of a model and estimate its generalization ability.

# Split defines the % of data that will be used in the training sample
# 1 - SPLIT = the % used for testing
SPLIT = 0.75 # Proportion of the data that will be used for training the model in a train-test split.



# --- 3) ALGORITHM CODE

# Load data
def load_data(path: str = "/path/to/csv/"):
    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target]) # drop() - remove a specified coumn from the DataFrame & return a new DataFrame
    y = data[target]                # Create a Series of DataFrame
    return X, y                     # Return as a tuple


#  Trains an algorithm using K-fold cross-validation and evaluates its performance by computing the mean absolute error.
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, # predictor variables
    y: pd.Series = None     # target variable
):

    accuracy = [] # Create a list that will store the accuracies of each fold

  
    for fold in range(0, K): # Loop to run K folds of cross-validation

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
 
# --- 4) MAIN FUNCTION
# Your algorithm code should contain modular code that can be run independently.
# You may want to include a final function that ties everything together, to allow
# the entire pipeline of loading the data and training the algorithm to be run all
# at once

# Execute training pipeline
def run():
    df = load_data() # Load the data first
    X, y = create_target_and_predictors(data=df) # Now split the data into predictors and target variables
    train_algorithm_with_cross_validation(X=X, y=y) # Finally, train the machine learning model
