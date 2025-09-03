# -*- coding: utf-8 -*-
"""
Diabetes Risk Prediction Analysis
=================================

Bu notebook diabetes dataset üzərində hərtərəfli data science analizi aparır.
Təqdim olunan tələblərə əsasən aşağıdakı metodları tətbiq edir:

1. Pandasla məlumatların düzgün oxunması
2. Statistik data types və vizuallaşdırmalar daxil olmaqla, datasetin xüsusiyyətlərini başa düşmək üçün EDA
3. Data Cleaning - null dəyərlərin yoxlanılması, onların düzgün və məntiqi idarə edilməsi
4. Data Preprocessing and Feature Engineering - Uyğun encoding üsullarından istifadə edilməsi
5. Data Preprocessing and Feature Engineering - scaling/normalizing numerical features
6. Feature Selection - Maşın öyrənmə modelinq üçün müvafiq featureları seçin
7. Splitting the Data - datasetin train və test setlərinə bölün
8. Düzgün maşın öyrənmə modelinin seçilməsi (mümkünsə 2 model)
9. Hər bir modelin test dataseti üzərində testi
10. Model Evaluation - Test dəstində müvəffəq ölçülərdən istifadə edərək model performansını qiymətləndirin
"""

# Lazımi kitabxanaları import edirik
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Matplotlib üçün Azərbaycan dili dəstəyi
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 60)
print("DIABETES RISK PREDICTION ANALYSIS")
print("=" * 60)

# 1. PANDASLA MƏLUMATLARIN DÜZGÜN OXUNMASI
print("\n1. MƏLUMATLAR YÜKLƏNIR...")
df = pd.read_csv('c:/Users/hp/CascadeProjects/DiseaseRiskPrediction/data/diabetes.csv')
print(f"✓ Dataset uğurla yükləndi: {df.shape[0]} sətir, {df.shape[1]} sütun")

# 2. STATISTIK DATA TYPES VƏ VİZUALLAŞDIRMALAR - EDA
print("\n2. EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 40)

# Dataset haqqında əsas məlumatlar
print("Dataset strukturu:")
print(df.info())
print("\nİlk 5 sətir:")
print(df.head())

print("\nStatistik xülasə:")
print(df.describe())

# Target dəyişəninin paylanması
print(f"\nTarget dəyişən (Outcome) paylanması:")
print(df['Outcome'].value_counts())
print(f"Diabetes olan: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
print(f"Diabetes olmayan: {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")

# Vizuallaşdırmalar
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Target dəyişəninin paylanması
axes[0,0].pie(df['Outcome'].value_counts(), labels=['Diabetes Yox', 'Diabetes Var'], 
              autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
axes[0,0].set_title('Diabetes Paylanması')

# Yaş paylanması
axes[0,1].hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,1].set_title('Yaş Paylanması')
axes[0,1].set_xlabel('Yaş')
axes[0,1].set_ylabel('Tezlik')

# BMI paylanması
axes[1,0].hist(df['BMI'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,0].set_title('BMI Paylanması')
axes[1,0].set_xlabel('BMI')
axes[1,0].set_ylabel('Tezlik')

# Glucose paylanması
axes[1,1].hist(df['Glucose'], bins=20, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].set_title('Glucose Paylanması')
axes[1,1].set_xlabel('Glucose')
axes[1,1].set_ylabel('Tezlik')

plt.tight_layout()
plt.show()

# Korelasiya matrisi
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Xüsusiyyətlər Arasında Korelasiya Matrisi')
plt.show()

# 3. DATA CLEANING - NULL DƏYƏRLƏRİN YOXLANILMASI
print("\n3. DATA CLEANING")
print("-" * 40)

print("Null dəyərlər:")
null_counts = df.isnull().sum()
print(null_counts)

if null_counts.sum() == 0:
    print("✓ Heç bir null dəyər tapılmadı!")
else:
    print("⚠ Null dəyərlər tapıldı və təmizlənəcək...")
    df = df.dropna()

# Sıfır dəyərlərin yoxlanılması (bəzi sahələrdə sıfır mənasız ola bilər)
print("\nSıfır dəyərlərin sayı:")
zero_counts = (df == 0).sum()
print(zero_counts)

# Bioloji olaraq sıfır ola bilməyən sahələri müəyyən edirik
biological_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
print(f"\nBioloji olaraq sıfır ola bilməyən sahələrdə sıfır dəyərlər:")
for field in biological_fields:
    if field in df.columns:
        zero_count = (df[field] == 0).sum()
        if zero_count > 0:
            print(f"{field}: {zero_count} sıfır dəyər")
            # Sıfır dəyərləri median ilə əvəz edirik
            median_val = df[df[field] != 0][field].median()
            df[field] = df[field].replace(0, median_val)
            print(f"  → {field} sahəsindəki sıfır dəyərlər median ({median_val:.2f}) ilə əvəz edildi")

print("✓ Data cleaning tamamlandı!")

# 4. DATA PREPROCESSING AND FEATURE ENGINEERING - ENCODING
print("\n4. DATA PREPROCESSING VƏ FEATURE ENGINEERING")
print("-" * 40)

# Yeni xüsusiyyətlər yaradırıq
print("Yeni xüsusiyyətlər yaradılır...")

# BMI kateqoriyaları
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal
    elif bmi < 30:
        return 2  # Overweight
    else:
        return 3  # Obese

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

# Yaş qrupları
def categorize_age(age):
    if age < 30:
        return 0  # Young
    elif age < 50:
        return 1  # Middle-aged
    else:
        return 2  # Senior

df['Age_Group'] = df['Age'].apply(categorize_age)

# Glucose səviyyəsi kateqoriyaları
def categorize_glucose(glucose):
    if glucose < 100:
        return 0  # Normal
    elif glucose < 126:
        return 1  # Prediabetes
    else:
        return 2  # Diabetes

df['Glucose_Category'] = df['Glucose'].apply(categorize_glucose)

print("✓ Yeni xüsusiyyətlər yaradıldı: BMI_Category, Age_Group, Glucose_Category")

# 5. SCALING/NORMALIZING NUMERICAL FEATURES
print("\n5. NUMERICAL FEATURES SCALING")
print("-" * 40)

# Numerical sütunları müəyyən edirik
numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Scaling üçün StandardScaler istifadə edirik
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("✓ Numerical xüsusiyyətlər StandardScaler ilə normallaşdırıldı")

# 6. FEATURE SELECTION
print("\n6. FEATURE SELECTION")
print("-" * 40)

# Bütün xüsusiyyətləri seçirik (target istisna olmaqla)
feature_columns = [col for col in df_scaled.columns if col != 'Outcome']
X = df_scaled[feature_columns]
y = df_scaled['Outcome']

# SelectKBest ilə ən yaxşı xüsusiyyətləri seçirik
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)

# Seçilmiş xüsusiyyətlərin adlarını alırıq
selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
print(f"Seçilmiş xüsusiyyətlər ({len(selected_features)}):")
for i, feature in enumerate(selected_features):
    score = selector.scores_[feature_columns.index(feature)]
    print(f"  {i+1}. {feature}: {score:.2f}")

# 7. SPLITTING THE DATA
print("\n7. DATA SPLITTING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data bölündü:")
print(f"  Training set: {X_train.shape[0]} nümunə")
print(f"  Test set: {X_test.shape[0]} nümunə")
print(f"  Training set diabetes nisbəti: {y_train.mean():.3f}")
print(f"  Test set diabetes nisbəti: {y_test.mean():.3f}")

# 8. MAŞIN ÖYRƏNMƏ MODELLƏRİNİN SEÇİLMƏSİ VƏ TƏLİMİ
print("\n8. MAŞIN ÖYRƏNMƏ MODELLƏRİ")
print("-" * 40)

# Model 1: Random Forest Classifier
print("Model 1: Random Forest Classifier təlim alır...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✓ Random Forest modeli təlim aldı")

# Model 2: Logistic Regression
print("Model 2: Logistic Regression təlim alır...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("✓ Logistic Regression modeli təlim aldı")

# 9. HƏR BİR MODELİN TEST DATASETİ ÜZƏRİNDƏ TESTİ
print("\n9. MODEL TESLƏRİ")
print("-" * 40)

# Random Forest təxminləri
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Logistic Regression təxminləri
lr_predictions = lr_model.predict(X_test)
lr_probabilities = lr_model.predict_proba(X_test)[:, 1]

print("✓ Hər iki model test dataseti üzərində təxmin etdi")

# 10. MODEL EVALUATION
print("\n10. MODEL PERFORMANS QİYMƏTLƏNDİRMƏSİ")
print("=" * 60)

def evaluate_model(y_true, y_pred, model_name):
    """Model performansını qiymətləndir"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Performansı:")
    print(f"  Accuracy (Dəqiqlik):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (Presizlik): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall (Həssaslıq):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:              {f1:.4f} ({f1*100:.2f}%)")
    
    return accuracy, precision, recall, f1

# Random Forest qiymətləndirməsi
rf_metrics = evaluate_model(y_test, rf_predictions, "Random Forest")

# Logistic Regression qiymətləndirməsi  
lr_metrics = evaluate_model(y_test, lr_predictions, "Logistic Regression")

# Confusion Matrix vizuallaşdırması
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest - Confusion Matrix')
axes[0].set_xlabel('Təxmin edilən')
axes[0].set_ylabel('Həqiqi')

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_predictions)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Logistic Regression - Confusion Matrix')
axes[1].set_xlabel('Təxmin edilən')
axes[1].set_ylabel('Həqiqi')

plt.tight_layout()
plt.show()

# Detailed Classification Report
print("\nƏTRAFLI KLASSİFİKASİYA HESABATI:")
print("\nRandom Forest:")
print(classification_report(y_test, rf_predictions, target_names=['Diabetes Yox', 'Diabetes Var']))

print("\nLogistic Regression:")
print(classification_report(y_test, lr_predictions, target_names=['Diabetes Yox', 'Diabetes Var']))

# Feature Importance (Random Forest üçün)
print("\nXÜSUSİYYƏTLƏRİN ƏHƏMİYYƏTİ (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Feature Importance vizuallaşdırması
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Xüsusiyyətlərin Əhəmiyyəti (Random Forest)')
plt.xlabel('Əhəmiyyət')
plt.tight_layout()
plt.show()

# Model müqayisəsi
print("\nMODEL MÜQAYİSƏSİ:")
print("-" * 40)
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [rf_metrics[0], lr_metrics[0]],
    'Precision': [rf_metrics[1], lr_metrics[1]],
    'Recall': [rf_metrics[2], lr_metrics[2]],
    'F1-Score': [rf_metrics[3], lr_metrics[3]]
})

print(comparison_df.round(4))

# Ən yaxşı modeli müəyyən edirik
best_model_idx = comparison_df['F1-Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\n🏆 Ən yaxşı model: {best_model_name}")
print(f"   F1-Score: {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}")

print("\n" + "=" * 60)
print("ANALİZ TAMAMLANDI!")
print("=" * 60)
print("\nXÜLASƏ:")
print("✓ Dataset uğurla yükləndi və təhlil edildi")
print("✓ Data cleaning və preprocessing aparıldı")
print("✓ Feature engineering və selection tətbiq edildi")
print("✓ İki müxtəlif maşın öyrənmə modeli təlim aldı")
print("✓ Modellər test edildi və performansları qiymətləndirildi")
print(f"✓ Ən yaxşı model: {best_model_name}")
