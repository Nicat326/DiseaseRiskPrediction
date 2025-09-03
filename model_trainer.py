import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Müxtəlif ML modellərini təlim edir"""
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        
        self.models['Logistic Regression'] = lr_model
        self.model_scores['Logistic Regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_prob),
            'report': classification_report(y_test, lr_pred, output_dict=True)
        }
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        self.models['Random Forest'] = rf_model
        self.model_scores['Random Forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'auc': roc_auc_score(y_test, rf_prob),
            'report': classification_report(y_test, rf_pred, output_dict=True)
        }
        
        # Ən yaxşı modeli seçirik
        best_auc = 0
        for name, scores in self.model_scores.items():
            if scores['auc'] > best_auc:
                best_auc = scores['auc']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"Ən yaxşı model: {self.best_model_name} (AUC: {best_auc:.3f})")
        
        return self.models, self.model_scores
    
    def save_models(self):
        """Modelleri saxlayır"""
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"{name} modeli saxlanıldı: {filename}")
    
    def load_models(self):
        """Saxlanılmış modelləri yükləyir"""
        model_files = {
            'Logistic Regression': 'models/logistic_regression_model.pkl',
            'Random Forest': 'models/random_forest_model.pkl'
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"{name} modeli yükləndi")
        
        # Ən yaxşı modeli təyin edirik (Random Forest default olaraq)
        if 'Random Forest' in self.models:
            self.best_model = self.models['Random Forest']
            self.best_model_name = 'Random Forest'
        elif 'Logistic Regression' in self.models:
            self.best_model = self.models['Logistic Regression']
            self.best_model_name = 'Logistic Regression'
    
    def predict_risk(self, input_data, model_name=None):
        """Xəstəlik riskini proqnozlaşdırır"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name, self.best_model)
        
        if model is None:
            raise ValueError("Model tapılmadı. Əvvəlcə modelləri təlim edin və ya yükləyin.")
        
        # Risk ehtimalını hesablayırıq
        risk_probability = model.predict_proba(input_data)[0, 1]
        risk_percentage = risk_probability * 100
        
        # Risk kateqoriyası
        if risk_percentage < 30:
            risk_category = "Aşağı Risk"
            risk_color = "green"
        elif risk_percentage < 60:
            risk_category = "Orta Risk"
            risk_color = "orange"
        else:
            risk_category = "Yüksək Risk"
            risk_color = "red"
        
        return {
            'risk_percentage': risk_percentage,
            'risk_category': risk_category,
            'risk_color': risk_color,
            'model_used': model_name,
            'confidence': max(risk_probability, 1 - risk_probability)
        }
    
    def get_model_performance(self):
        """Model performans məlumatlarını qaytarır"""
        return self.model_scores
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Feature importance qaytarır (Random Forest üçün)"""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
        return None
