import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class ModelingAnalysis:
    def __init__(self, global_rankings_url, target_column='spi', test_size=0.2, random_state=42):
        self.df = self.load_data(global_rankings_url)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.pipeline = self.create_pipeline()

    def load_data(self, url):
        return pd.read_csv(url)

    def prepare_data(self):
    
        columns_to_drop = [self.target_column, 'rank', 'prev_rank', 'name', 'league']
        existing_columns = set(self.df.columns)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]
    
        X = self.df.drop(columns=columns_to_drop, errors='ignore')
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def create_pipeline(self):
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        return pipeline

    def train_model(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline.named_steps['regressor']

    def evaluate_model(self):
        """
        Evaluates the model's performance on the test set using mean squared error and R squared metrics.
    
        Returns
        -------
        tuple
            A tuple containing mean squared error and R squared score of the model's predictions.
        """
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        return mse, r2

    def plot_rank_importance(self):
        feature_indices = [self.X_train.columns.get_loc('rank'), self.X_train.columns.get_loc('prev_rank')]
        importances = self.model.feature_importances_[feature_indices]
        feature_names = ['rank', 'prev_rank']

        plt.bar(feature_names, importances)
        plt.title('Rank and Previous Rank Importance')
        plt.show()

    
