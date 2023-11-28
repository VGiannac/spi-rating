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

    def scatterplot_spi_ratings(self):
        """
        Create a scatter plot comparing SPI ratings (spi1 vs spi2).

        Displays the scatter plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='spi1', y='spi2', data=self.data, alpha=0.3)
        plt.title('SPI Ratings Comparison')
        plt.xlabel('SPI Team 1')
        plt.ylabel('SPI Team 2')
        plt.show()

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