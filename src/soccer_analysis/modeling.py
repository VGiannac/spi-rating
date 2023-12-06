import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

    def ModelingAnalysis(data_url):
        df = pd.read_csv(data_url)
    
        # Assuming you have a column 'outcome' with values 'win', 'draw', 'loss'
        df['outcome'] = df['outcome'].map({'win': 1, 'draw': 0, 'loss': -1})
    
        features = ['spi1', 'spi2', 'proj_score1', 'proj_score2']
        target = 'outcome'
    
        X = df[features]
        y = df[target]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_classifier(X_train, y_train):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        return clf
    
    def evaluate_classifier(clf, X_test, y_test):
        y_pred = clf.predict(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
    
        print(classification_report(y_test, y_pred))


