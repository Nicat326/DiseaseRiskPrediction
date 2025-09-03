# Xəstəlik Riski Hesablama və Tövsiyə Sistemi
## Disease Risk Calculation and Recommendation System

Bu layihə istifadəçilərin sağlamlıq məlumatlarına əsasən xəstəlik riskini hesablayan və tövsiyələr verən interaktiv bir sistemdir.

## Xüsusiyyətlər

- **Risk Hesablama**: Machine Learning modelləri ilə xəstəlik riskini faizlə göstərir
- **Explainability**: SHAP values ilə riskə ən çox təsir edən faktorları vizuallaşdırır
- **Simulyasiya**: Müxtəlif parametrləri dəyişərək riskin necə dəyişəcəyini görün
- **İstifadəçi Dostu İnterfeys**: Streamlit ilə hazırlanmış sadə və effektiv veb interfeys

## Quraşdırma

1. Layihəni klonlayın və ya yükləyin
2. Virtual mühit yaradın:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Tələb olunan paketləri quraşdırın:
```bash
pip install -r requirements.txt
```

4. Tətbiqi işə salın:
```bash
streamlit run app.py
```

## Layihə Strukturu

```
DiseaseRiskPrediction/
├── app.py                 # Əsas Streamlit tətbiqi
├── data_loader.py         # Dataset yükləmə və hazırlama
├── model_trainer.py       # ML modellərinin təlimi
├── explainer.py          # SHAP explainability
├── simulator.py          # Risk simulyasiyası
├── chatbot.py            # Sağlamlıq chatbotu
├── utils.py              # Köməkçi funksiyalar
├── models/               # Təlim edilmiş modellər
├── data/                 # Dataset faylları
└── requirements.txt      # Python tələbləri
```

## Diabetes Risk Prediction Analysis

Bu layihə diabetes dataset üzərində hərtərəfli data science analizi aparır və risk proqnozlaşdırma modelləri yaradır.

## Xüsusiyyətlər

✅ **Pandasla məlumatların düzgün oxunması**  
✅ **Exploratory Data Analysis (EDA)** - statistik təhlil və vizuallaşdırmalar  
✅ **Data Cleaning** - null və problemli dəyərlərin idarə edilməsi  
✅ **Data Preprocessing** - encoding və feature engineering  
✅ **Feature Scaling** - numerical xüsusiyyətlərin normallaşdırılması  
✅ **Feature Selection** - ən əhəmiyyətli xüsusiyyətlərin seçilməsi  
✅ **Data Splitting** - train/test setlərinin yaradılması  
✅ **Machine Learning** - 2 müxtəlif modelin təlimi  
✅ **Model Testing** - test dataseti üzərində qiymətləndirmə  
✅ **Model Evaluation** - hərtərəfli performans analizi  

## Quraşdırma

1. Layihəni klonlayın və ya yükləyin
2. Virtual mühit yaradın:
```bash
python -m venv diabetes_env
diabetes_env\Scripts\activate  # Windows
```

3. Lazımi paketləri quraşdırın:
```bash
pip install -r requirements.txt
```

## İstifadə

### Python Script olaraq
```bash
python diabetes_analysis.py
```

### Jupyter Notebook-da
1. Jupyter Notebook-u işə salın:
```bash
jupyter notebook
```

2. Yeni notebook yaradın və `diabetes_analysis.py` faylının məzmununu kopyalayın

3. Hər bir hissəni ayrı cell-lərdə icra edin

## Dataset

Dataset `data/diabetes.csv` faylında yerləşir və aşağıdakı xüsusiyyətləri ehtiva edir:

- **Pregnancies**: Hamiləlik sayı
- **Glucose**: Glucose səviyyəsi
- **BloodPressure**: Qan təzyiqi
- **SkinThickness**: Dəri qalınlığı
- **Insulin**: İnsulin səviyyəsi
- **BMI**: Bədən kütləsi indeksi
- **DiabetesPedigreeFunction**: Diabetes ailə tarixi
- **Age**: Yaş
- **Outcome**: Hədəf dəyişən (0: Diabetes yox, 1: Diabetes var)

## Modellər

Layihədə 2 maşın öyrənmə modeli istifadə olunur:

1. **Random Forest Classifier** - Ensemble learning metodu
2. **Logistic Regression** - Klassik klassifikasiya metodu

## Nəticələr

Analiz aşağıdakı nəticələri təqdim edir:

- Ətraflı EDA və vizuallaşdırmalar
- Data cleaning və preprocessing hesabatı
- Feature importance analizi
- Model performans müqayisəsi
- Confusion matrix və classification report
- Ən yaxşı modelin müəyyən edilməsi

## Fayllar

- `diabetes_analysis.py` - Əsas analiz skripti
- `data/diabetes.csv` - Dataset
- `requirements.txt` - Lazımi paketlər
- `README.md` - Bu faylın özü

## Qeydlər

- Bütün mətnlər Azərbaycan dilindədir
- Kod həm Python script, həm də Jupyter Notebook formatında işləyir
- Vizuallaşdırmalar avtomatik olaraq göstərilir
- Nəticələr ətraflı şəkildə çap edilir

## İstifadə

1. Tətbiqi açın və şəxsi məlumatlarınızı daxil edin
2. Risk hesablama nəticəsini görün
3. Riskə təsir edən faktorları analiz edin
4. Simulyasiya ilə müxtəlif ssenariləri yoxlayın
5. Chatbot ilə sağlamlıq tövsiyələri alın

## Texnologiyalar

- **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest)
- **Explainability**: SHAP
- **Vizualizasiya**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit
- **Dataset**: Pima Indians Diabetes Dataset

