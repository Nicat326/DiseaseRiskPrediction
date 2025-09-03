import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelExplainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self):
        """SHAP explainer-i başladır"""
        try:
            # Random Forest üçün TreeExplainer
            if hasattr(self.model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Logistic Regression üçün LinearExplainer
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
            
            print("SHAP explainer uğurla başladıldı")
        except Exception as e:
            print(f"SHAP explainer başladılarkən xəta: {e}")
            # Fallback olaraq Permutation explainer
            self.explainer = shap.Explainer(self.model.predict, self.X_train)
    
    def explain_prediction(self, input_data):
        """Tək bir proqnoz üçün SHAP values hesablayır"""
        try:
            if self.explainer is None:
                self.initialize_explainer()
            
            if self.explainer is None:
                print("SHAP explainer yaradıla bilmədi")
                return None
            
            # Input data formatını düzəldirik
            if hasattr(input_data, 'values'):
                input_data = input_data.values
            
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            # SHAP values hesablayırıq
            if hasattr(self.model, 'estimators_'):
                # Random Forest üçün
                shap_values = self.explainer.shap_values(input_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # Logistic Regression üçün
                shap_values = self.explainer.shap_values(input_data)
            
            # Formatı düzəldirik
            if hasattr(shap_values, 'shape'):
                if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analizi xətası: {e}")
            # Fallback - sadə feature importance qaytarırıq
            try:
                if hasattr(self.model, 'feature_importances_'):
                    return self.model.feature_importances_
                elif hasattr(self.model, 'coef_'):
                    return np.abs(self.model.coef_[0])
                else:
                    return np.zeros(len(self.feature_names))
            except:
                return None
    
    def create_feature_importance_plot(self, input_data, shap_values):
        """Feature importance plotunu yaradır"""
        if shap_values is None:
            return None
        
        # SHAP values-ları absolute dəyərlərinə görə sıralayırıq
        importance_data = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Value': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            'Input_Value': input_data[0] if len(input_data.shape) > 1 else input_data
        })
        
        importance_data['Abs_SHAP'] = abs(importance_data['SHAP_Value'])
        importance_data = importance_data.sort_values('Abs_SHAP', ascending=True)
        
        # Plotly ilə interaktiv qrafik
        fig = go.Figure()
        
        colors = ['red' if x < 0 else 'green' for x in importance_data['SHAP_Value']]
        
        fig.add_trace(go.Bar(
            y=importance_data['Feature'],
            x=importance_data['SHAP_Value'],
            orientation='h',
            marker_color=colors,
            text=[f'{val:.3f}' for val in importance_data['SHAP_Value']],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Riskə Təsir Edən Faktorlar (SHAP Values)',
            xaxis_title='Riskə Təsir (SHAP Value)',
            yaxis_title='Faktorlar',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_waterfall_plot(self, input_data, shap_values, base_value=None):
        """Waterfall plot yaradır"""
        if shap_values is None:
            return None
        
        if base_value is None:
            base_value = 0.5  # Default base probability
        
        # SHAP values-ları sıralayırıq
        feature_impacts = list(zip(self.feature_names, shap_values[0] if len(shap_values.shape) > 1 else shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Waterfall plot məlumatları
        features = [f[0] for f in feature_impacts]
        values = [f[1] for f in feature_impacts]
        
        # Kumulativ dəyərlər
        cumulative = [base_value]
        for val in values:
            cumulative.append(cumulative[-1] + val)
        
        fig = go.Figure()
        
        # Base value
        fig.add_trace(go.Bar(
            x=['Base Risk'],
            y=[base_value],
            name='Base Risk',
            marker_color='lightblue'
        ))
        
        # Feature contributions
        for i, (feature, value) in enumerate(feature_impacts):
            color = 'red' if value > 0 else 'green'
            fig.add_trace(go.Bar(
                x=[feature],
                y=[abs(value)],
                base=[cumulative[i] if value > 0 else cumulative[i] - abs(value)],
                name=f'{feature}: {value:.3f}',
                marker_color=color
            ))
        
        # Final prediction
        fig.add_trace(go.Bar(
            x=['Final Risk'],
            y=[cumulative[-1]],
            name='Final Risk',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Risk Hesablama Prosesi (Waterfall)',
            xaxis_title='Faktorlar',
            yaxis_title='Risk Ehtimalı',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def get_top_risk_factors(self, input_data, shap_values, top_n=3):
        """Ən yüksək risk faktorlarını qaytarır"""
        if shap_values is None:
            return []
        
        feature_impacts = list(zip(self.feature_names, shap_values[0] if len(shap_values.shape) > 1 else shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for i in range(min(top_n, len(feature_impacts))):
            feature, impact = feature_impacts[i]
            direction = "artırır" if impact > 0 else "azaldır"
            top_factors.append({
                'feature': feature,
                'impact': impact,
                'direction': direction,
                'magnitude': abs(impact)
            })
        
        return top_factors
    
    def create_feature_distribution_plot(self, feature_name, current_value, data_sample=None):
        """Müəyyən feature üçün paylanma qrafiki"""
        if data_sample is None:
            data_sample = self.X_train
        
        feature_idx = self.feature_names.index(feature_name)
        feature_data = data_sample[:, feature_idx]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=feature_data,
            nbinsx=30,
            name='Paylanma',
            opacity=0.7
        ))
        
        # İstifadəçinin dəyəri
        fig.add_vline(
            x=current_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Sizin dəyəriniz: {current_value:.2f}"
        )
        
        fig.update_layout(
            title=f'{feature_name} Paylanması',
            xaxis_title=feature_name,
            yaxis_title='Tezlik',
            height=400
        )
        
        return fig
