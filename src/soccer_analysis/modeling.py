import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ModelingAnalysis:

    def __init__(self, csvfile):
        self.csvfile = csvfile
        self.df = pd.read_csv(csvfile)
        self.features = ['league', 'home_team', 'away_team']
        self.target = 'winning_percentage'

    def clean_data(self):
        # assuming 'winning_percentage' is a calculated field in the dataframe
        # we need to add the field 'winning_percentage' to the dataframe
        self.df['winning_percentage'] = self.df.apply(lambda row: self.calculate_winning_percentage(row), axis=1)
        self.df.dropna(inplace=True)

    def calculate_winning_percentage(self, row):
        # helper function to calculate the winning percentage
        total_games = row['total_games']
        total_wins = row['total_wins']
        return total_wins / total_games if total_games != 0 else 0

    def prepare_data(self):
        self.df = pd.get_dummies(self.df, columns=self.features)
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        return X, y

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def make_predictions(self, X_new):
        return self.model.predict(X_new)

