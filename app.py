import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_loader import DataLoader
from model_trainer import ModelTrainer
from explainer import ModelExplainer
from simulator import RiskSimulator
import os

# Streamlit konfiqurasiyası
st.set_page_config(
    page_title="🏥 Diabet Risk Proqnozu",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Möhtəşəm və maraqlı dizayn
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Orbitron:wght@400;700;900&display=swap');
    
    /* Ana tema - Futuristik */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        animation: gradientWave 8s ease infinite;
        color: white;
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
        position: relative;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        animation: gradientWave 8s ease infinite;
        color: white;
    }
    
    @keyframes gradientWave {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Başlıq - Neon effekti */
    .main-header {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 400% 400%;
        animation: rainbowShift 3s ease infinite;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Orbitron', monospace;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.3),
            0 0 60px rgba(255, 107, 107, 0.4),
            inset 0 0 30px rgba(255,255,255,0.1);
        border: 2px solid rgba(255,255,255,0.2);
        text-shadow: 0 0 20px rgba(255,255,255,0.8);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rainbowShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Risk kartları - Neon glow */
    .risk-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        border: 2px solid;
        margin: 2rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
    }
    
    .risk-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .risk-card:hover::before {
        left: 100%;
    }
    
    .risk-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 35px 70px rgba(0,0,0,0.4);
    }
    
    .risk-high {
        border-color: #ff6b6b;
        box-shadow: 
            0 25px 50px rgba(0,0,0,0.3),
            0 0 50px rgba(255, 107, 107, 0.5);
    }
    
    .risk-high:hover {
        box-shadow: 
            0 35px 70px rgba(0,0,0,0.4),
            0 0 80px rgba(255, 107, 107, 0.8);
    }
    
    .risk-medium {
        border-color: #feca57;
        box-shadow: 
            0 25px 50px rgba(0,0,0,0.3),
            0 0 50px rgba(254, 202, 87, 0.5);
    }
    
    .risk-medium:hover {
        box-shadow: 
            0 35px 70px rgba(0,0,0,0.4),
            0 0 80px rgba(254, 202, 87, 0.8);
    }
    
    .risk-low {
        border-color: #4ecdc4;
        box-shadow: 
            0 25px 50px rgba(0,0,0,0.3),
            0 0 50px rgba(78, 205, 196, 0.5);
    }
    
    .risk-low:hover {
        box-shadow: 
            0 35px 70px rgba(0,0,0,0.4),
            0 0 80px rgba(78, 205, 196, 0.8);
    }
    
    .risk-card h2 {
        color: white;
        font-size: 3rem;
        margin: 1rem 0;
        font-weight: 800;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 30px currentColor;
    }
    
    .risk-card h3 {
        color: white;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .risk-card p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Sidebar - Glassmorphism */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 2px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important;
    }
    
    .css-1d391kg h2, .css-1d391kg label {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .css-1d391kg h2 {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.5rem !important;
        text-align: center !important;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 2rem !important;
    }
    
    /* Düymələr - Neon */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #feca57) !important;
        background-size: 300% 300% !important;
        animation: buttonGradient 3s ease infinite !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    @keyframes buttonGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4) !important;
    }
    
    /* Input sahələri - Glow */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4ecdc4 !important;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.5) !important;
    }
    
    /* Tabs - Modern */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
        color: white !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
    }
    
    /* Mesajlar - Glow */
    .stAlert {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        border-radius: 15px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2) !important;
    }
    
    .stSuccess {
        border-color: #4ecdc4 !important;
        box-shadow: 0 15px 30px rgba(78, 205, 196, 0.3) !important;
    }
    
    .stError {
        border-color: #ff6b6b !important;
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.3) !important;
    }
    
    .stInfo {
        border-color: #45b7d1 !important;
        box-shadow: 0 15px 30px rgba(69, 183, 209, 0.3) !important;
    }
    
    /* Mətnlər */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Statistik kartları */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(25px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.05), transparent);
        animation: rotate 8s linear infinite;
    }
    
    .stat-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 30px 60px rgba(0,0,0,0.3);
    }
    
    .stat-card h3 {
        color: #4ecdc4;
        font-size: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .stat-card h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: 800;
        font-family: 'Orbitron', monospace;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 20px currentColor;
    }
    
    .stat-card p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        position: relative;
        z-index: 1;
    }
    
    /* Plotly qrafiklər */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(25px) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        box-shadow: 0 0 30px rgba(78, 205, 196, 0.7);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Məlumatları yükləyir və hazırlayır"""
    data_loader = DataLoader()
    data_loader.load_pima_diabetes_data()
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    return data_loader, X_train, X_test, y_train, y_test

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Modelləri təlim edir və saxlayır"""
    model_trainer = ModelTrainer()
    
    # Əgər saxlanılmış modellər varsa, onları yükləyirik
    if os.path.exists('models/random_forest_model.pkl'):
        model_trainer.load_models()
    else:
        model_trainer.train_models(X_train, X_test, y_train, y_test)
        model_trainer.save_models()
    
    return model_trainer

def create_stat_card(title, value, icon, description=""):
    """Sadə statistik kartı yaradır"""
    return f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <h3>{title}</h3>
        <h2>{value}</h2>
        {f'<p style="font-size: 0.9rem; color: #718096; margin-top: 0.5rem;">{description}</p>' if description else ''}
    </div>
    """

def main():
    # Başlıq
    st.markdown('<div class="main-header">🏥 Xəstəlik Riski Hesablama Sistemi</div>', 
                unsafe_allow_html=True)
    
    # Məlumatları yükləyirik
    try:
        data_loader, X_train, X_test, y_train, y_test = load_and_prepare_data()
        model_trainer = train_models(X_train, X_test, y_train, y_test)
        
        # Explainer və simulator yaradırıq
        explainer = ModelExplainer(model_trainer.best_model, X_train, data_loader.get_feature_names())
        
        # SHAP explainer-i dərhal başladırıq
        st.info("🔄 SHAP analizi hazırlanır...")
        explainer.initialize_explainer()
        
        if explainer.explainer is not None:
            st.success("✅ SHAP analizi hazırdır!")
        else:
            st.warning("⚠️ SHAP analizi hazırlanmadı, sadə analiz istifadə ediləcək")
        
        simulator = RiskSimulator(model_trainer, data_loader)
        
    except Exception as e:
        st.error(f"🚨 Sistem yüklənərkən xəta baş verdi: {e}")
        st.info("💡 Səhifəni yeniləyin və ya modelləri yenidən təlim edin")
        return
    
    # Sidebar - İstifadəçi məlumatları
    with st.sidebar:
        st.markdown("## 📋 Şəxsi Məlumatlarınız")
        st.markdown("---")
        
        feature_descriptions = data_loader.get_feature_descriptions()
        user_input = {}
        
        # İstifadəçi girişləri
        user_input['Pregnancies'] = st.number_input(
            f"👶 {feature_descriptions['Pregnancies']}", 
            min_value=0, max_value=15, value=1, step=1
        )
        
        user_input['Glucose'] = st.number_input(
            f"🩸 {feature_descriptions['Glucose']} (mg/dL)", 
            min_value=70, max_value=200, value=120, step=5
        )
        
        user_input['BloodPressure'] = st.number_input(
            f"💓 {feature_descriptions['BloodPressure']} (mmHg)", 
            min_value=60, max_value=140, value=80, step=5
        )
        
        user_input['SkinThickness'] = st.number_input(
            f"📏 {feature_descriptions['SkinThickness']} (mm)", 
            min_value=0, max_value=50, value=20, step=1
        )
        
        user_input['Insulin'] = st.number_input(
            f"💉 {feature_descriptions['Insulin']} (μU/mL)", 
            min_value=0, max_value=500, value=80, step=10
        )
        
        user_input['BMI'] = st.number_input(
            f"⚖️ {feature_descriptions['BMI']}", 
            min_value=15.0, max_value=50.0, value=25.0, step=0.1
        )
        
        user_input['DiabetesPedigreeFunction'] = st.number_input(
            f"🧬 {feature_descriptions['DiabetesPedigreeFunction']}", 
            min_value=0.0, max_value=2.5, value=0.5, step=0.1
        )
        
        user_input['Age'] = st.number_input(
            f"🎂 {feature_descriptions['Age']}", 
            min_value=20, max_value=80, value=30, step=1
        )
        
        st.markdown("---")
        # Risk hesablama düyməsi
        if st.button("🔍 Riski Hesabla", type="primary"):
            st.session_state.calculate_risk = True
    
    # Əsas məzmun
    if hasattr(st.session_state, 'calculate_risk') and st.session_state.calculate_risk:
        
        # Risk hesablama
        try:
            user_input_scaled = data_loader.scale_input(user_input)
            risk_result = model_trainer.predict_risk(user_input_scaled)
            
            # Risk nəticəsi göstərmə - sadə və aydın
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                risk_class = "risk-low" if risk_result['risk_percentage'] < 30 else \
                           "risk-medium" if risk_result['risk_percentage'] < 60 else "risk-high"
                
                risk_icon = "✅" if risk_result['risk_percentage'] < 30 else \
                           "⚠️" if risk_result['risk_percentage'] < 60 else "🚨"
                
                st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{risk_icon}</div>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">
                        {risk_result['risk_percentage']:.1f}%
                    </h1>
                    <h2 style="font-size: 1.5rem; margin: 0.5rem 0;">
                        {risk_result['risk_category']}
                    </h2>
                    <p style="margin-top: 1rem; font-size: 1rem;">
                        İstifadə olunan model: {risk_result['model_used']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sadə statistikalar
            st.markdown("---")
            st.markdown("## 📊 Sizin Göstəricilərinizdə")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bmi_status = "Normal" if 18.5 <= user_input['BMI'] < 25 else \
                           "Artıq çəki" if 25 <= user_input['BMI'] < 30 else "Piylənmə"
                bmi_desc = f"BMI: {user_input['BMI']:.1f}"
                st.markdown(create_stat_card("BMI Kateqoriyası", bmi_status, "⚖️", bmi_desc), 
                           unsafe_allow_html=True)
            
            with col2:
                glucose_status = "Normal" if user_input['Glucose'] < 100 else "Yüksək"
                glucose_desc = f"{user_input['Glucose']} mg/dL"
                st.markdown(create_stat_card("Qan Şəkəri", glucose_status, "🩸", glucose_desc), 
                           unsafe_allow_html=True)
            
            with col3:
                bp_status = "Normal" if user_input['BloodPressure'] < 80 else "Yüksək"
                bp_desc = f"{user_input['BloodPressure']} mmHg"
                st.markdown(create_stat_card("Qan Təzyiqi", bp_status, "💓", bp_desc), 
                           unsafe_allow_html=True)
            
            with col4:
                age_desc = f"{user_input['Age']} yaş"
                st.markdown(create_stat_card("Yaş Qrupu", 
                           "Gənc" if user_input['Age'] < 35 else "Orta yaş" if user_input['Age'] < 55 else "Yaşlı", 
                           "🎂", age_desc), 
                           unsafe_allow_html=True)
            
            # Tabs
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs([
                "📊 Ətraflı Analiz", 
                "🔄 Simulyasiya", 
                "📈 Model Performansı"
            ])
            
            with tab1:
                st.markdown("## 📊 Riskə Təsir Edən Faktorlar")
                
                # SHAP analizi
                try:
                    # Əvvəlcə explainer-in hazır olduğunu yoxlayırıq
                    if hasattr(explainer, 'explainer') and explainer.explainer is not None:
                        shap_values = explainer.explain_prediction(user_input_scaled)
                        
                        if shap_values is not None and len(shap_values) > 0:
                            # Feature importance plot
                            fig_importance = explainer.create_feature_importance_plot(user_input_scaled, shap_values)
                            if fig_importance:
                                st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Top risk factors
                            top_factors = explainer.get_top_risk_factors(user_input_scaled, shap_values)
                            
                            if top_factors:
                                st.markdown("### 🔝 Ən Təsirli Faktorlar")
                                for i, factor in enumerate(top_factors, 1):
                                    direction_icon = "📈" if factor['direction'] == "artırır" else "📉"
                                    factor_desc = feature_descriptions.get(factor['feature'], factor['feature'])
                                    
                                    st.markdown(f"""
                                    <div class="info-card">
                                        <h4>{i}. {direction_icon} {factor_desc}</h4>
                                        <p>Riski <strong>{factor['direction']}</strong> (təsir dərəcəsi: {factor['impact']:.3f})</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("📊 Risk faktorları analiz edilir...")
                        else:
                            # Fallback - model feature importance göstər
                            st.warning("⚠️ SHAP analizi hazırlanır. Bu vaxt model əsaslı analiz göstərilir:")
                            if hasattr(model_trainer.best_model, 'feature_importances_'):
                                importance_data = {
                                    'Faktor': data_loader.get_feature_names(),
                                    'Əhəmiyyət': model_trainer.best_model.feature_importances_
                                }
                                import plotly.express as px
                                fig = px.bar(
                                    importance_data, 
                                    x='Əhəmiyyət', 
                                    y='Faktor',
                                    title='Model Feature Importance',
                                    orientation='h'
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("🚨 SHAP explainer hazırlanmadı. Sistem yenidən başladılır...")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"🚨 Analiz xətası: {str(e)}")
                    st.info("💡 Sistem modelləri yenidən yükləyir...")
                    # Fallback - sadə model importance göstər
                    try:
                        if hasattr(model_trainer.best_model, 'feature_importances_'):
                            importance_data = {
                                'Faktor': data_loader.get_feature_names(),
                                'Əhəmiyyət': model_trainer.best_model.feature_importances_
                            }
                            import plotly.express as px
                            fig = px.bar(
                                importance_data, 
                                x='Əhəmiyyət', 
                                y='Faktor',
                                title='Model Feature Importance (Fallback)',
                                orientation='h'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("📊 Analiz məlumatları hazırlanır...")
                
                # Tövsiyələr
                recommendations = simulator.get_improvement_recommendations(user_input)
                if recommendations:
                    st.markdown("### 💡 Yaxşılaşdırma Tövsiyələri")
                    for rec in recommendations:
                        priority_icon = "🔴" if rec['priority'] == 'Yüksək' else "🟡"
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>{priority_icon} {rec['factor']}</h4>
                            <p>{rec['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("## 🔄 Risk Simulyasiyası")
                st.markdown("Müxtəlif dəyişikliklər etdikdə riskinizin necə dəyişəcəyini görün:")
                
                # Default ssenariləri yaradırıq
                scenarios = simulator.create_default_scenarios(user_input)
                
                if scenarios:
                    simulation_results = simulator.simulate_risk_changes(user_input, scenarios)
                    
                    # Simulyasiya qrafiki
                    fig_sim = simulator.create_simulation_plot(simulation_results)
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Simulyasiya cədvəli
                    st.markdown("### 📋 Ssenari Təfərrüatları")
                    sim_df = pd.DataFrame(simulation_results)
                    st.dataframe(sim_df[['scenario', 'risk_percentage', 'risk_category', 'changes']], 
                               use_container_width=True)
                
                # İnteraktiv slider simulyasiyası
                st.markdown("### 🎛️ İnteraktiv Simulyasiya")
                selected_feature = st.selectbox(
                    "Hansı faktoru dəyişdirmək istəyirsiniz?",
                    options=list(feature_descriptions.keys()),
                    format_func=lambda x: feature_descriptions[x]
                )
                
                if selected_feature:
                    fig_slider = simulator.create_interactive_slider_simulation(user_input, selected_feature)
                    if fig_slider:
                        st.plotly_chart(fig_slider, use_container_width=True)
            
            with tab3:
                st.markdown("## 📈 Model Performansı")
                
                # Model performans məlumatları
                model_scores = model_trainer.get_model_performance()
                
                if model_scores:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Dəqiqlik Göstəriciləri")
                        for model_name, scores in model_scores.items():
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>🤖 {model_name}</h4>
                                <p>📊 Dəqiqlik: <strong>{scores['accuracy']:.3f}</strong></p>
                                <p>📈 AUC: <strong>{scores['auc']:.3f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📊 Model Müqayisəsi")
                        model_names = list(model_scores.keys())
                        accuracies = [model_scores[name]['accuracy'] for name in model_names]
                        aucs = [model_scores[name]['auc'] for name in model_names]
                        
                        fig_performance = go.Figure()
                        fig_performance.add_trace(go.Bar(
                            name='Dəqiqlik',
                            x=model_names,
                            y=accuracies,
                            marker_color='#3498db'
                        ))
                        fig_performance.add_trace(go.Bar(
                            name='AUC',
                            x=model_names,
                            y=aucs,
                            marker_color='#2980b9'
                        ))
                        
                        fig_performance.update_layout(
                            title='Model Performans Müqayisəsi',
                            yaxis_title='Göstərici Dəyəri',
                            barmode='group',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_performance, use_container_width=True)
                
                # Dataset məlumatları
                st.markdown("### 📋 Dataset Məlumatları")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(create_stat_card("Ümumi Nümunə", f"{len(data_loader.data)}", "📊"), 
                               unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_stat_card("Təlim Nümunələri", f"{len(X_train)}", "🎓"), 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_stat_card("Test Nümunələri", f"{len(X_test)}", "🧪"), 
                               unsafe_allow_html=True)
                
                with col4:
                    st.markdown(create_stat_card("Feature Sayı", f"{len(data_loader.get_feature_names())}", "🔢"), 
                               unsafe_allow_html=True)
                
                # Dataset preview
                if st.checkbox("📋 Dataset nümunəsini göstər"):
                    st.dataframe(data_loader.data.head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"🚨 Risk hesablanarkən xəta baş verdi: {e}")
    
    else:
        # İlk səhifə məlumatları
        st.markdown("""
        <div class="info-card">
            <h2>🎯 Sistem Haqqında</h2>
            <p>Bu sistem sizin sağlamlıq məlumatlarınıza əsasən <strong>diabetes riski</strong>ni hesablayır və sizə şəxsi tövsiyələr verir.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Xüsusiyyətlər
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>✨ Əsas Xüsusiyyətlər</h3>
                <p>🤖 <strong>Machine Learning</strong>: Dəqiq risk hesablaması</p>
                <p>📊 <strong>Explainability</strong>: Riskə təsir edən faktorların analizi</p>
                <p>🔄 <strong>Simulyasiya</strong>: "Əgər" ssenariləri</p>
                <p>📈 <strong>Vizualizasiya</strong>: İnteraktiv qrafiklər</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>🚀 Necə İstifadə Etmək Olar</h3>
                <p>1️⃣ Sol paneldə şəxsi məlumatları daxil edin</p>
                <p>2️⃣ "Riski Hesabla" düyməsinə basın</p>
                <p>3️⃣ Nəticələri analiz edin və tövsiyələri oxuyun</p>
                <p>4️⃣ Simulyasiya ilə müxtəlif ssenariləri yoxlayın</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Nümunə məlumatlar
        st.markdown("### 📝 Nümunə Risk Profilləri")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card" style="border-left-color: #38a169;">
                <h4>🟢 Aşağı Risk Profili</h4>
                <p>👤 Yaş: 25</p>
                <p>⚖️ BMI: 22</p>
                <p>🩸 Qan şəkəri: 90 mg/dL</p>
                <p>💓 Qan təzyiqi: 70 mmHg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card" style="border-left-color: #dd6b20;">
                <h4>🟡 Orta Risk Profili</h4>
                <p>👤 Yaş: 45</p>
                <p>⚖️ BMI: 28</p>
                <p>🩸 Qan şəkəri: 120 mg/dL</p>
                <p>💓 Qan təzyiqi: 85 mmHg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card" style="border-left-color: #e53e3e;">
                <h4>🔴 Yüksək Risk Profili</h4>
                <p>👤 Yaş: 55</p>
                <p>⚖️ BMI: 35</p>
                <p>🩸 Qan şəkəri: 160 mg/dL</p>
                <p>💓 Qan təzyiqi: 95 mmHg</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Vacib qeyd
        st.markdown("""
        <div class="info-card" style="border-left-color: #e53e3e;">
            <h3>⚠️ Vacib Qeyd</h3>
            <p>Bu sistem yalnız <strong>informativ məqsədlər</strong> üçündür. Dəqiq diaqnoz və müalicə üçün həkiminizlə məsləhətləşin.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
