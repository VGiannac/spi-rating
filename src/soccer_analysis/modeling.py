import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelingAnalysis:
    def __init__(self, global_rankings_url, target_column='rank_class', test_size=0.2, random_state=42):
        self.global_rankings_url = global_rankings_url
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.df = self.load_data()
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def load_data(self):
        return pd.read_csv(self.global_rankings_url)

    def prepare_data(self):
        X = self.df.drop(columns=[self.target_column, 'rank', 'prev_rank', 'name', 'league'])
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def create_pipeline(self):
        return RandomForestClassifier(random_state=self.random_state)

    def train_model(self):
        pipeline = self.create_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        self.model = pipeline

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f'Accuracy: {accuracy}')

