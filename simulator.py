import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class RiskSimulator:
    def __init__(self, model_trainer, data_loader):
        self.model_trainer = model_trainer
        self.data_loader = data_loader
        self.feature_names = data_loader.get_feature_names()
        self.feature_descriptions = data_loader.get_feature_descriptions()
        
    def simulate_risk_changes(self, original_input, modifications):
        """Müxtəlif dəyişikliklər üçün risk dəyişikliklərini simulyasiya edir"""
        results = []
        
        # Orijinal risk
        original_scaled = self.data_loader.scale_input(original_input)
        original_risk = self.model_trainer.predict_risk(original_scaled)
        
        results.append({
            'scenario': 'Hazırkı Vəziyyət',
            'risk_percentage': original_risk['risk_percentage'],
            'risk_category': original_risk['risk_category'],
            'changes': 'Heç bir dəyişiklik'
        })
        
        # Hər bir dəyişiklik üçün simulyasiya
        for scenario_name, changes in modifications.items():
            modified_input = original_input.copy()
            change_description = []
            
            for feature, new_value in changes.items():
                if feature in modified_input:
                    old_value = modified_input[feature]
                    modified_input[feature] = new_value
                    
                    # Dəyişiklik təsvirini hazırlayırıq
                    feature_desc = self.feature_descriptions.get(feature, feature)
                    if new_value > old_value:
                        change_description.append(f"{feature_desc}: {old_value:.1f} → {new_value:.1f} (↑)")
                    else:
                        change_description.append(f"{feature_desc}: {old_value:.1f} → {new_value:.1f} (↓)")
            
            # Yeni riski hesablayırıq
            modified_scaled = self.data_loader.scale_input(modified_input)
            modified_risk = self.model_trainer.predict_risk(modified_scaled)
            
            results.append({
                'scenario': scenario_name,
                'risk_percentage': modified_risk['risk_percentage'],
                'risk_category': modified_risk['risk_category'],
                'changes': '; '.join(change_description),
                'risk_change': modified_risk['risk_percentage'] - original_risk['risk_percentage']
            })
        
        return results
    
    def create_default_scenarios(self, original_input):
        """Default simulyasiya ssenariləri yaradır"""
        scenarios = {}
        
        # Sağlam həyat tərzi ssenarisi
        healthy_changes = {}
        if 'BMI' in original_input and original_input['BMI'] > 25:
            healthy_changes['BMI'] = max(22, original_input['BMI'] - 5)
        if 'Glucose' in original_input and original_input['Glucose'] > 100:
            healthy_changes['Glucose'] = max(90, original_input['Glucose'] - 20)
        if 'BloodPressure' in original_input and original_input['BloodPressure'] > 80:
            healthy_changes['BloodPressure'] = max(70, original_input['BloodPressure'] - 10)
        
        if healthy_changes:
            scenarios['Sağlam Həyat Tərzi'] = healthy_changes
        
        # Çəki itkisi ssenarisi
        if 'BMI' in original_input and original_input['BMI'] > 23:
            scenarios['10% Çəki İtkisi'] = {
                'BMI': original_input['BMI'] * 0.9
            }
        
        # Qan şəkəri nəzarəti
        if 'Glucose' in original_input and original_input['Glucose'] > 100:
            scenarios['Qan Şəkəri Nəzarəti'] = {
                'Glucose': min(100, original_input['Glucose'] - 30)
            }
        
        # Risk artırıcı ssenari
        risk_increase = {}
        if 'Age' in original_input:
            risk_increase['Age'] = original_input['Age'] + 10
        if 'BMI' in original_input:
            risk_increase['BMI'] = original_input['BMI'] + 5
        
        scenarios['10 İl Sonra (dəyişiklik olmasa)'] = risk_increase
        
        return scenarios
    
    def create_simulation_plot(self, simulation_results):
        """Simulyasiya nəticələri üçün qrafik yaradır"""
        df = pd.DataFrame(simulation_results)
        
        # Risk dəyişikliklərini rənglərlə göstəririk
        colors = []
        for _, row in df.iterrows():
            if row['scenario'] == 'Hazırkı Vəziyyət':
                colors.append('blue')
            elif 'risk_change' in row and row['risk_change'] < 0:
                colors.append('green')  # Risk azalması
            elif 'risk_change' in row and row['risk_change'] > 0:
                colors.append('red')    # Risk artması
            else:
                colors.append('gray')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['scenario'],
            y=df['risk_percentage'],
            marker_color=colors,
            text=[f"{val:.1f}%" for val in df['risk_percentage']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<br><extra></extra>'
        ))
        
        fig.update_layout(
            title='Müxtəlif Ssenarilərdə Risk Dəyişikliyi',
            xaxis_title='Ssenari',
            yaxis_title='Risk Faizi (%)',
            height=500,
            showlegend=False
        )
        
        # Risk kateqoriyalarını göstərmək üçün horizontal xətlər
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Aşağı Risk Hədi")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Yüksək Risk Hədi")
        
        return fig
    
    def create_interactive_slider_simulation(self, original_input, feature_name):
        """Müəyyən feature üçün interaktiv slider simulyasiyası"""
        if feature_name not in original_input:
            return None
        
        original_value = original_input[feature_name]
        
        # Feature üçün uyğun range təyin edirik
        ranges = {
            'BMI': (15, 50, 0.5),
            'Glucose': (70, 200, 5),
            'BloodPressure': (60, 140, 5),
            'Age': (20, 80, 1),
            'Pregnancies': (0, 15, 1),
            'Insulin': (0, 500, 10),
            'SkinThickness': (0, 50, 1),
            'DiabetesPedigreeFunction': (0, 2.5, 0.1)
        }
        
        if feature_name not in ranges:
            return None
        
        min_val, max_val, step = ranges[feature_name]
        
        # Müxtəlif dəyərlər üçün risk hesablayırıq
        test_values = np.arange(min_val, max_val + step, step)
        risks = []
        
        for test_val in test_values:
            test_input = original_input.copy()
            test_input[feature_name] = test_val
            test_scaled = self.data_loader.scale_input(test_input)
            risk_result = self.model_trainer.predict_risk(test_scaled)
            risks.append(risk_result['risk_percentage'])
        
        # Qrafik yaradırıq
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_values,
            y=risks,
            mode='lines+markers',
            name='Risk Əyrisi',
            line=dict(color='blue', width=2)
        ))
        
        # Hazırkı dəyəri vurğulayırıq
        current_risk_idx = np.argmin(np.abs(test_values - original_value))
        fig.add_trace(go.Scatter(
            x=[original_value],
            y=[risks[current_risk_idx]],
            mode='markers',
            name='Hazırkı Dəyər',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f'{self.feature_descriptions.get(feature_name, feature_name)} Dəyişikliyinin Riskə Təsiri',
            xaxis_title=self.feature_descriptions.get(feature_name, feature_name),
            yaxis_title='Risk Faizi (%)',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def get_improvement_recommendations(self, original_input, top_n=3):
        """Risk azaltmaq üçün tövsiyələr verir"""
        recommendations = []
        
        # BMI tövsiyəsi
        if 'BMI' in original_input and original_input['BMI'] > 25:
            target_bmi = 24
            weight_loss = ((original_input['BMI'] - target_bmi) / original_input['BMI']) * 100
            recommendations.append({
                'factor': 'Bədən Kütlə İndeksi (BMI)',
                'current': original_input['BMI'],
                'target': target_bmi,
                'recommendation': f"BMI-ni {target_bmi}-ə qədər azaltın (təxminən {weight_loss:.1f}% çəki itkisi)",
                'priority': 'Yüksək'
            })
        
        # Qan şəkəri tövsiyəsi
        if 'Glucose' in original_input and original_input['Glucose'] > 100:
            target_glucose = 100
            recommendations.append({
                'factor': 'Qan Şəkəri',
                'current': original_input['Glucose'],
                'target': target_glucose,
                'recommendation': f"Qan şəkərini {target_glucose} mg/dL-ə qədər azaltın",
                'priority': 'Yüksək'
            })
        
        # Qan təzyiqi tövsiyəsi
        if 'BloodPressure' in original_input and original_input['BloodPressure'] > 80:
            target_bp = 80
            recommendations.append({
                'factor': 'Qan Təzyiqi',
                'current': original_input['BloodPressure'],
                'target': target_bp,
                'recommendation': f"Qan təzyiqini {target_bp} mmHg-ə qədər azaltın",
                'priority': 'Orta'
            })
        
        return recommendations[:top_n]
