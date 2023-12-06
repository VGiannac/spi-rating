## __init__.py

"""soccer_analysis Package

This package provides classes for analyzing soccer match data.

Classes:
    - DataSummary: Summary and basic analysis of soccer match data.
    - SoccerAnalysis: Exploratory data analysis on soccer match data.
    - InferenceAnalysis: Statistical inference analysis on soccer match data.
    - ModelingAnalysis: Modeling and predictions on soccer match data.

Usage:
    You can use the classes in this package to perform various analyses on soccer match data.

Example:
    To create an instance of SoccerAnalysis:
    ```
    from soccer_analysis import SoccerAnalysis

    data_url = 'https://raw.githubusercontent.com/VGiannac/soccer_analysis/main/spi_matches_latest.csv'
    soccer_analysis_instance = SoccerAnalysis(data_url)
    soccer_analysis_instance.scatterplot_spi_ratings()
    ```
"""
