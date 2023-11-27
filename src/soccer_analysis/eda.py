import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class SoccerAnalysis:
    def __init__(self, data):
        self.data = data

    def numerical_descriptive_statistics(self):
        return self.data.describe()

    def scatterplot_spi_ratings(self):
        sns.scatterplot(x='spi1', y='spi2', data=self.data)
        plt.title('SPI Ratings Comparison')
        plt.show()

    def histogram_projected_scores(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['proj_score1'], kde=True, bins=20, color='blue', label='Projected Score Team 1')
        sns.histplot(self.data['proj_score2'], kde=True, bins=20, color='orange', label='Projected Score Team 2')
        plt.title('Distribution of Projected Scores')
        plt.legend()
        plt.show()

    def correlation_matrix(self):
        correlation_matrix = self.data[['spi1', 'spi2', 'proj_score1', 'proj_score2', 'score1', 'score2']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
