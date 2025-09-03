#!/usr/bin/env python3
"""
Modelləri təlim etmək üçün ayrı skript
Bu skripti işə salmaqla modelləri əvvəlcədən təlim edə bilərsiniz
"""

from data_loader import DataLoader
from model_trainer import ModelTrainer
import os

def main():
    print("🚀 Xəstəlik Riski Proqnozlaşdırma Modellərinin Təlimi Başlayır...")
    
    # Məlumatları yükləyirik
    print("📊 Məlumatlar yüklənir...")
    data_loader = DataLoader()
    data_loader.load_pima_diabetes_data()
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    print(f"✅ Məlumatlar hazırlandı:")
    print(f"   - Təlim nümunələri: {len(X_train)}")
    print(f"   - Test nümunələri: {len(X_test)}")
    print(f"   - Feature sayı: {len(data_loader.get_feature_names())}")
    
    # Modelləri təlim edirik
    print("\n🤖 Modellər təlim edilir...")
    model_trainer = ModelTrainer()
    models, scores = model_trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Nəticələri göstəririk
    print("\n📈 Model Performansı:")
    for model_name, score_dict in scores.items():
        print(f"   {model_name}:")
        print(f"     - Dəqiqlik: {score_dict['accuracy']:.3f}")
        print(f"     - AUC: {score_dict['auc']:.3f}")
    
    # Modelləri saxlayırıq
    print("\n💾 Modellər saxlanır...")
    model_trainer.save_models()
    
    print("\n✅ Təlim tamamlandı! İndi Streamlit tətbiqini işə sala bilərsiniz:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
