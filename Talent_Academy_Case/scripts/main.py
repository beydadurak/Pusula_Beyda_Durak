import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "Talent_Academy_Case_DT_2025.xlsx"

data = pd.read_excel(DATA_PATH)
print(data.head())

print("Shape:", data.shape)

print("\nInfo:")
print(data.info())

print("\nDescribe:")
print(data.describe(include="all").T)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicated rows:", data.duplicated().sum())

data = data.drop_duplicates().reset_index(drop=True)
print("Yeni shape:", data.shape)

data["TedaviSuresi_Sayisal"] = data["TedaviSuresi"].str.extract(r"(\d+)").astype(float)

data["UygulamaSuresi_Dakika"] = data["UygulamaSuresi"].str.extract(r"(\d+)").astype(float)

def normalize_gender(x):
    if pd.isna(x): return "Bilinmiyor"
    s = str(x).lower().strip()
    if s in ["kadın", "kadin", "k", "female", "f"]: return "Kadın"
    if s in ["erkek", "e", "male", "m"]: return "Erkek"
    return "Bilinmiyor"

data["Cinsiyet_Norm"] = data["Cinsiyet"].map(normalize_gender)

def normalize_blood(x):
    if pd.isna(x): return "Bilinmiyor"
    s = str(x).strip().upper()
    return s

data["KanGrubu_Norm"] = data["KanGrubu"].map(normalize_blood)

data.head()

def count_items(x):
    if pd.isna(x) or str(x).strip() == "":
        return 0
    return len([i for i in str(x).split(",") if i.strip() != ""])

for col in ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri", "Bolum"]:
    data[f"{col}_Sayisi"] = data[col].map(count_items)

data[["KronikHastalik", "KronikHastalik_Sayisi",
      "Alerji", "Alerji_Sayisi",
      "Tanilar", "Tanilar_Sayisi"]].head(10)
sns.set_theme(style="whitegrid")

num_cols = [
    "Yas", "KronikHastalik_Sayisi", "Alerji_Sayisi", "Tanilar_Sayisi",
    "UygulamaYerleri_Sayisi", "Bolum_Sayisi", "UygulamaSuresi_Dakika",
    "TedaviSuresi_Sayisal"
]
num_cols = [c for c in num_cols if c in data.columns]


plt.figure(figsize=(10, 5))
sns.heatmap(
    data[num_cols].isna(),
    cmap="Reds",
    cbar=False
)
plt.title("Eksik Veri Isı Haritası")
plt.show()


for c in num_cols:
    plt.figure(figsize=(5, 3))
    sns.histplot(data[c], kde=True)
    plt.title(f"{c} dağılımı")
    plt.show()

corr = data[num_cols].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Korelasyon Isı Haritası")
plt.show()


data[[ "KronikHastalik","KronikHastalik_Sayisi",
       "Alerji","Alerji_Sayisi",
       "Tanilar","Tanilar_Sayisi",
       "UygulamaYerleri","UygulamaYerleri_Sayisi",
       "Bolum","Bolum_Sayisi"]].head(5)


y = data["TedaviSuresi_Sayisal"].astype(float)

feature_cols = [
    "Yas",
    "KronikHastalik_Sayisi", "Alerji_Sayisi", "Tanilar_Sayisi",
    "UygulamaYerleri_Sayisi", "Bolum_Sayisi",
    "UygulamaSuresi_Dakika",
    "Cinsiyet_Norm", "KanGrubu_Norm", "Uyruk", "TedaviAdi"
]

X = data[feature_cols].copy()

mask = y.notna()
X = X[mask]
y = y[mask]

numeric_features = [
    "Yas", "KronikHastalik_Sayisi", "Alerji_Sayisi", "Tanilar_Sayisi",
    "UygulamaYerleri_Sayisi", "Bolum_Sayisi", "UygulamaSuresi_Dakika"
]

categorical_features = ["Cinsiyet_Norm", "KanGrubu_Norm", "Uyruk", "TedaviAdi"]

X.head()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_ready = preprocess.fit_transform(X)
print(X_ready.shape)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_ready, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("DataFrame kolonları:", data.columns.tolist())
