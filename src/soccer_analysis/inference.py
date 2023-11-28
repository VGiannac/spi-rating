# src/soccer_analysis/inference.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class InferenceAnalysis:
    def __init__(self, data_url):
        """
        Constructor for the SoccerAnalysis class.

        Args:
            data_url (str): The file path or URL for the data that has to be loaded.
        """
        self.data = pd.read_csv(data_url)

    def scatterplot_spi_ratings_hexbin(self):
        """
        Create a jointplot with hexbin for SPI ratings.

        Displays the jointplot.
        """
        sns.set(style="white", color_codes=True)
        g = sns.jointplot(x='spi1', y='spi2', data=self.data, kind='hex', color='blue', height=8)
        g.set_axis_labels('SPI Team 1', 'SPI Team 2')
        plt.suptitle('SPI Ratings Hexbin Comparison', y=1.02)
        plt.show()

    def boxplot_match_outcomes(self):
        """
        Create boxplots for match outcomes (score1 and score2).

        Displays the boxplots.
        """
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='variable', y='value', data=pd.melt(self.data[['score1', 'score2']]))
        plt.title('Match Outcomes Boxplots')
        plt.xlabel('Match Outcome')
        plt.ylabel('Score')
        plt.show()

    def distplot_probabilities(self):
        """
        Create distribution plots for win probabilities (prob1 and prob2).

        Displays the distribution plots.
        """
        plt.figure(figsize=(12, 8))
        sns.histplot(self.data[['prob1', 'prob2']], kde=True, bins=20, alpha=0.5)
        plt.title('Win Probabilities Distribution')
        plt.xlabel('Win Probability')
        plt.ylabel('Frequency')
        plt.show()