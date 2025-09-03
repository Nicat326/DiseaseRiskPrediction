import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_pima_diabetes_data(self):
        """Pima Indians Diabetes Dataset yükləyir"""
        # Əgər lokal faylda yoxdursa, sample data yaradırıq
        if not os.path.exists('data/diabetes.csv'):
            self._create_sample_data()
        
        self.data = pd.read_csv('data/diabetes.csv')
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        return self.data
    
    def _create_sample_data(self):
        """Sample diabetes dataset yaradır"""
        os.makedirs('data', exist_ok=True)
        
        # Pima Indians Diabetes Dataset-ə bənzər sample data
        np.random.seed(42)
        n_samples = 768
        
        data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
            'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 120),
            'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 50),
            'Insulin': np.random.normal(80, 100, n_samples).clip(0, 800),
            'BMI': np.random.normal(32, 7, n_samples).clip(15, 50),
            'DiabetesPedigreeFunction': np.random.gamma(0.5, 1, n_samples).clip(0, 2.5),
            'Age': np.random.normal(33, 12, n_samples).clip(21, 81),
        }
        
        # Target dəyişəni yaradırıq (diabetes riski)
        risk_score = (
            data['Glucose'] * 0.01 +
            data['BMI'] * 0.05 +
            data['Age'] * 0.02 +
            data['Pregnancies'] * 0.1 +
            np.random.normal(0, 1, n_samples)
        )
        
        data['Outcome'] = (risk_score > np.percentile(risk_score, 65)).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv('data/diabetes.csv', index=False)
        
    def prepare_data(self):
        """Məlumatları ML üçün hazırlayır"""
        if self.data is None:
            self.load_pima_diabetes_data()
            
        # Features və target ayırırıq
        X = self.data[self.feature_names]
        y = self.data['Outcome']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardization
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def get_feature_names(self):
        """Feature adlarını qaytarır"""
        return self.feature_names
    
    def get_feature_descriptions(self):
        """Feature təsvirləri"""
        descriptions = {
            'Pregnancies': 'Hamiləlik sayı',
            'Glucose': 'Qan şəkəri səviyyəsi',
            'BloodPressure': 'Qan təzyiqi',
            'SkinThickness': 'Dəri qalınlığı',
            'Insulin': 'İnsulin səviyyəsi',
            'BMI': 'Bədən kütlə indeksi',
            'DiabetesPedigreeFunction': 'Diabetes ailə tarixi',
            'Age': 'Yaş'
        }
        return descriptions
    
    def scale_input(self, input_data):
        """Yeni input məlumatını scale edir"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
            return self.scaler.transform(input_df[self.feature_names])
        else:
            return self.scaler.transform(input_data)
