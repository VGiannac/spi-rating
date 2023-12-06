import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ModelingAnalysis:
    def __init__(self, data_url):
        self.data = pd.read_csv(data_url)
        self.df = pd.read_csv(data_url)
        self.target = 'spi1'

    def filter_chinese_league_2019(self):
        self.df = self.df[(self.df['league'] == 'Chinese Super League') & (self.df['season'] == 2019)]
        self.df.reset_index(drop=True, inplace=True)

    def prepare_data(self):
        self.features = ['league_id', 'league', 'team1', 'team2', 'spi2', 'prob1', 'prob2']
        X = self.df[self.features]
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


