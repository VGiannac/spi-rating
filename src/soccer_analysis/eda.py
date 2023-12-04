# src/soccer_analysis/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class SoccerAnalysis:
    """
    A class for performing exploratory data analysis on soccer match data.
    """

    def __init__(self, data_url):
        """
        Constructor for the SoccerAnalysis class.

        Args:
            data_url (str): The file path or URL for the data that has to be loaded.
        """
        self.data = pd.read_csv(data_url)
        self.model = None  # Initialize the model attribute

    def display_data_head(self):
        """
        Display the first few rows of the loaded data.

        Returns:
        - pd.DataFrame: The first few rows of the data.
        """
        return self.data.head()

    def numerical_descriptive_statistics(self):
        """
        Compute numerical descriptive statistics for the dataset.

        Returns:
        - pd.DataFrame: Numerical descriptive statistics.
        """
        return self.data.describe()

    def scatterplot_spi_ratings(self):
        """
        Create a scatter plot comparing SPI ratings (spi1 vs spi2).

        Displays the scatter plot.
        """
        sns.scatterplot(x='spi1', y='spi2', data=self.data)
        plt.title('SPI Ratings Comparison')
        plt.show()

    def histogram_projected_scores(self):
        """
        Create a histogram for projected scores of both teams.

        Displays the histogram.
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['proj_score1'], kde=True, bins=20, color='blue', label='Projected Score Team 1')
        sns.histplot(self.data['proj_score2'], kde=True, bins=20, color='orange', label='Projected Score Team 2')
        plt.title('Distribution of Projected Scores')
        plt.legend()
        plt.show()

    def correlation_matrix(self):
        """
        Create a correlation matrix for selected columns.

        Displays the correlation matrix heatmap.
        """
        correlation_matrix = self.data[['spi1', 'spi2', 'proj_score1', 'proj_score2', 'score1', 'score2']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

    def prepare_data_for_classification(self):
        """
        Prepare data for classification.

        Assign labels (1 for team1 wins, 2 for team2 wins, 0 for draw) based on match outcomes.
        """
        # You can adjust this based on your specific criteria for win/draw/loss
        self.data['result'] = self.data.apply(lambda row: 1 if row['score1'] > row['score2'] else (2 if row['score1'] < row['score2'] else 0), axis=1)

    def train_regression_model(self):
        """
        Train a regression model to predict match scores.

        Returns:
        - sklearn.linear_model.LinearRegression: The trained regression model.
        """
        # Select relevant features and target variable
        features = self.data[['spi1', 'spi2', 'proj_score1', 'proj_score2']]
        target = self.data[['score1', 'score2']]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Initialize and train the regression model
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        # Evaluate the model
        predictions = regression_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        # Set the trained model to the class attribute
        self.model = regression_model

        return regression_model

    def predict_match_outcomes(self, features):
        """
        Predict match outcomes using the trained classification model.

        Args:
        - features (pd.DataFrame): Features for which to predict match outcomes.

        Returns:
        - pd.DataFrame: Predicted match outcomes.
        """
        if self.model is None:
            print('Model not trained. Call train_classification_model() first.')
            return

        return pd.DataFrame(self.model.predict(features), columns=['predicted_result'])

    


