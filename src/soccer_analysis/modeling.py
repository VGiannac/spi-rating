import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelingAnalysis:
    def __init__(self, data_url):
        self.data_url = data_url
        self.df = pd.read_csv(data_url)
        self.features = ['spi1', 'spi2', 'proj_score1', 'proj_score2']
        self.target = 'outcome'
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()

    def preprocess_data(self):
        # Assuming you have a column 'outcome' with values 'win', 'draw', 'loss'
        self.df['outcome'] = self.df['outcome'].map({'win': 1, 'draw': 0, 'loss': -1})

        X = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_classifier(self):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(self.X_train, self.y_train)
        return clf

    def evaluate_classifier(self, clf):
        y_pred = clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        print(classification_report(self.y_test, y_pred))
