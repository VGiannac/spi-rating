# __main__.py file

from soccer_analysis.summary import DataSummary
from soccer_analysis.eda import SoccerAnalysis
from soccer_analysis.inference import InferenceAnalysis
from soccer_analysis.modeling import ModelingAnalysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
filterwarnings("ignore")

def main():
    # Define the data URL
    data_url = 'https://raw.githubusercontent.com/VGiannac/soccer_analysis/main/spi_matches_latest.csv'
    # data_url = 'https://raw.githubusercontent.com/VGiannac/soccer_analysis/main/spi_global_rankings.csv'

    # Create instances of your analysis classes
    matches_summary = DataSummary(data_url)
    # global_rankings_summary = DataSummary(global_rankings_data_url)
    soccer_analysis_instance = SoccerAnalysis(data_url)
    soccer_inference_instance = InferenceAnalysis(data_url)
    modeling_analysis = ModelingAnalysis(data_url) 

    # Example usage of the methods
    matches_summary.explain_head_tail()
    soccer_analysis_instance.scatterplot_spi_ratings()
    soccer_inference_instance.scatterplot_spi_ratings_hexbin()
    

main()


