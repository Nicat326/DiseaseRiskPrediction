import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

class Utils:
    """Köməkçi funksiyalar sinfi"""
    
    @staticmethod
    def create_risk_gauge(risk_percentage, title="Risk Göstəricisi"):
        """Risk faizi üçün gauge chart yaradır"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def format_risk_message(risk_percentage, risk_category):
        """Risk mesajını formatlaşdırır"""
        if risk_percentage < 30:
            emoji = "✅"
            message = "Təbriklər! Sizin diabetes riskiniz aşağıdır."
        elif risk_percentage < 60:
            emoji = "⚠️"
            message = "Diqqət! Orta səviyyədə diabetes riskiniz var."
        else:
            emoji = "🚨"
            message = "Təcili! Yüksək diabetes riskiniz var. Həkiminizlə məsləhətləşin!"
        
        return f"{emoji} {message}"
    
    @staticmethod
    def calculate_bmi_category(bmi):
        """BMI kateqoriyasını hesablayır"""
        if bmi < 18.5:
            return "Çatışmazlıq", "underweight"
        elif bmi < 25:
            return "Normal", "normal"
        elif bmi < 30:
            return "Artıq çəki", "overweight"
        else:
            return "Piylənmə", "obese"
    
    @staticmethod
    def get_health_tips():
        """Sağlamlıq tövsiyələri qaytarır"""
        tips = [
            "🥗 Gündə ən azı 5 porsiya meyvə və tərəvəz yeyin",
            "🚶‍♂️ Gündə 10,000 addım atmağa çalışın",
            "💧 Gündə 8-10 stəkan su için",
            "😴 Gecədə 7-9 saat yuxu alın",
            "🧘‍♀️ Stress idarəetməsi texnikalarını öyrənin",
            "🚭 Siqarət çəkməyin və alkohol məhdudlaşdırın",
            "⚖️ Çəkinizi müntəzəm yoxlayın",
            "🩺 İllik sağlamlıq yoxlaması keçirin"
        ]
        return tips
    
    @staticmethod
    def create_feature_correlation_heatmap(data, feature_names):
        """Feature korrelyasiya heatmap yaradır"""
        correlation_matrix = data[feature_names].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Korrelyasiyası",
            color_continuous_scale="RdBu"
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def save_user_report(user_input, risk_result, recommendations, filename=None):
        """İstifadəçi hesabatını saxlayır"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_report_{timestamp}.txt"
        
        os.makedirs('reports', exist_ok=True)
        filepath = os.path.join('reports', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== XƏSTƏLİK RİSKİ HESABATI ===\n\n")
            f.write(f"Tarix: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n")
            
            f.write("ŞƏXSI MƏLUMATLAR:\n")
            for key, value in user_input.items():
                f.write(f"- {key}: {value}\n")
            
            f.write(f"\nRİSK NƏTİCƏSİ:\n")
            f.write(f"- Risk faizi: {risk_result['risk_percentage']:.1f}%\n")
            f.write(f"- Risk kateqoriyası: {risk_result['risk_category']}\n")
            f.write(f"- İstifadə olunan model: {risk_result['model_used']}\n")
            
            if recommendations:
                f.write(f"\nTÖVSİYƏLƏR:\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec['factor']}: {rec['recommendation']}\n")
            
            f.write(f"\n⚠️ QEYD: Bu hesabat yalnız informativ məqsədlər üçündür. "
                   f"Dəqiq diaqnoz üçün həkiminizlə məsləhətləşin.\n")
        
        return filepath
    
    @staticmethod
    def validate_input_ranges(user_input):
        """İstifadəçi girişlərini yoxlayır"""
        ranges = {
            'Pregnancies': (0, 20),
            'Glucose': (50, 250),
            'BloodPressure': (40, 200),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (10, 70),
            'DiabetesPedigreeFunction': (0, 5),
            'Age': (18, 120)
        }
        
        warnings = []
        for feature, value in user_input.items():
            if feature in ranges:
                min_val, max_val = ranges[feature]
                if value < min_val or value > max_val:
                    warnings.append(f"{feature} dəyəri qeyri-adi görünür: {value}")
        
        return warnings
    
    @staticmethod
    def create_age_risk_distribution(data):
        """Yaş qruplarına görə risk paylanması"""
        age_groups = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], 
                           labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        
        risk_by_age = data.groupby(age_groups)['Outcome'].mean() * 100
        
        fig = go.Figure(data=[
            go.Bar(x=risk_by_age.index, y=risk_by_age.values,
                   marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title='Yaş Qruplarına Görə Risk Paylanması',
            xaxis_title='Yaş Qrupu',
            yaxis_title='Risk Faizi (%)',
            height=400
        )
        
        return fig
    
    @staticmethod
    def get_azerbaijani_translations():
        """Azərbaycan dilində tərcümələr"""
        translations = {
            'Pregnancies': 'Hamiləlik sayı',
            'Glucose': 'Qan şəkəri səviyyəsi',
            'BloodPressure': 'Qan təzyiqi',
            'SkinThickness': 'Dəri qalınlığı',
            'Insulin': 'İnsulin səviyyəsi',
            'BMI': 'Bədən kütlə indeksi',
            'DiabetesPedigreeFunction': 'Diabetes ailə tarixi',
            'Age': 'Yaş',
            'Low Risk': 'Aşağı Risk',
            'Medium Risk': 'Orta Risk',
            'High Risk': 'Yüksək Risk'
        }
        return translations
