# src/soccer_analysis/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    
    


