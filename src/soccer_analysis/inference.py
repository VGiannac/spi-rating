## soccer_analysis/inference.py

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

        Displays the jointplot using Seaborn and Matplotlib.
        """
        sns.set(style="white", color_codes=True)

        # Seaborn Version
        g = sns.jointplot(x='spi1', y='spi2', data=self.data, kind='hex', color='blue', height=8)
        g.set_axis_labels('SPI Team 1', 'SPI Team 2')
        plt.suptitle('SPI Ratings Hexbin Comparison (Seaborn)', y=1.02)
        plt.show()

        # Matplotlib Version
        plt.figure(figsize=(8, 8))
        plt.hexbin(self.data['spi1'], self.data['spi2'], gridsize=20, cmap='Blues')
        plt.title('SPI Ratings Hexbin Comparison (Matplotlib)')
        plt.xlabel('SPI Team 1')
        plt.ylabel('SPI Team 2')
        plt.show()

    def boxplot_match_outcomes(self):
        """
        Create boxplots for match outcomes (score1 and score2).

        Displays the boxplots using Seaborn and Matplotlib.
        """
        # Seaborn Version
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='variable', y='value', data=pd.melt(self.data[['score1', 'score2']]))
        plt.title('Match Outcomes Boxplots (Seaborn)')
        plt.xlabel('Match Outcome')
        plt.ylabel('Score')
        plt.show()

        # Matplotlib Version
        plt.figure(figsize=(12, 8))
        plt.boxplot([self.data['score1'], self.data['score2']], labels=['Score 1', 'Score 2'])
        plt.title('Match Outcomes Boxplots (Matplotlib)')
        plt.xlabel('Match Outcome')
        plt.ylabel('Score')
        plt.show()

    def distplot_probabilities(self):
        """
        Create distribution plots for win probabilities (prob1 and prob2).

        Displays the distribution plots using Seaborn and Matplotlib.
        """
        # Seaborn Version
        plt.figure(figsize=(12, 8))
        sns.histplot(self.data[['prob1', 'prob2']], kde=True, bins=20, alpha=0.5)
        plt.title('Win Probabilities Distribution (Seaborn)')
        plt.xlabel('Win Probability')
        plt.ylabel('Frequency')
        plt.show()

        # Matplotlib Version
        plt.figure(figsize=(12, 8))
        plt.hist([self.data['prob1'], self.data['prob2']], bins=20, alpha=0.5, label=['Prob1', 'Prob2'])
        plt.title('Win Probabilities Distribution (Matplotlib)')
        plt.xlabel('Win Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


