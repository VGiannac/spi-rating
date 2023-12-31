import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class ModelingAnalysis:
    """
        Initialize the ModelingAnalysis class.

        Parameters
        ----------
        global_rankings_url : str
            The URL or file path to the CSV containing global rankings data.
        target_column : str, optional
            The target column for prediction, default is 'spi'.
        test_size : float, optional
            The proportion of the dataset to include in the test split, default is 0.2.
        random_state : int, optional
            Controls the randomness of the data splitting, default is 42.
        """
    def __init__(self, global_rankings_url, target_column='spi', test_size=0.2, random_state=42):
        self.df = self.load_data(global_rankings_url)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.pipeline = self.create_pipeline()

    def load_data(self, url):
        """
        Load data from a CSV file.

        Parameters
        ----------
        url : str
            The URL or file path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame.
        """
        return pd.read_csv(url)

    def prepare_data(self):
        """
        Prepare the data for training and testing.

        Returns
        -------
        tuple
            A tuple containing the training and testing sets for features and target variable.
        """
        columns_to_drop = [self.target_column, 'rank', 'prev_rank', 'name', 'league']
        existing_columns = set(self.df.columns)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]
    
        X = self.df.drop(columns=columns_to_drop, errors='ignore')
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def create_pipeline(self):
        """
        Create a data processing and modeling pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The constructed pipeline.
        """
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        return pipeline

    def train_model(self):
         """
        Train the model using the prepared data.
        """
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline.named_steps['regressor']

    def evaluate_model(self):
        """
        Evaluates the model's performance on the test set using mean squared error and R squared metrics.
    
        Returns
        -------
        tuple
            A tuple containing mean squared error and R squared score of the model's predictions.
        """
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        return mse, r2

    def plot_predictions(self):
        """
        Plots a scatter plot of predicted values against actual values on the test set.
        """
        predictions = self.pipeline.predict(self.X_test)

        plt.scatter(self.y_test, predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

    
