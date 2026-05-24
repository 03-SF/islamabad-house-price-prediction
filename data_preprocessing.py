"""
DATA PREPROCESSING SCRIPT - IMPROVED VERSION
=============================================
Key improvements over original:
1. Area converted to numeric Marla (not arbitrary label encoding)
2. Corrupt price rows (Arab/very low values) filtered out
3. Location grouped by broad area to reduce noise from 115 unique values
4. Binary amenity columns added for Drawing/Dining/Study/Prayer rooms
5. Property age feature engineered from Built Year
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv('zameen_results.csv')

# ── Remove junk rows (Twitter/Gmail share links) ──────────────────────────
df = df[df['URL'].str.contains('zameen.com/Property/', na=False)]
df = df[~df['URL'].str.contains('twitter.com|mail.google.com', na=False)]
print(f"\n📊 After removing junk rows: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n" + "=" * 70)
print("STEP 2: HANDLE MISSING VALUES & CORRUPT PRICES")
print("=" * 70)

# Drop rows with missing price
df = df.dropna(subset=['Price (PKR)'])

# ── Filter corrupt prices ──────────────────────────────────────────────────
# Some prices like "1 Arab" were saved as 1.0, 1.4, 1.35, 1.5, 1.75
# Real prices are always > 500,000 (5 Lakh minimum)
df = df[df['Price (PKR)'] > 500_000]
# Remove extreme outliers (above 2 billion - likely data errors)
df = df[df['Price (PKR)'] < 2_000_000_000]
print(f"✅ After price filtering: {df.shape[0]} rows")
print(f"   Price range: PKR {df['Price (PKR)'].min():,.0f} → PKR {df['Price (PKR)'].max():,.0f}")

print("\n" + "=" * 70)
print("STEP 3: REMOVE DUPLICATES")
print("=" * 70)

before = len(df)
df = df.drop_duplicates(subset=['URL'])
print(f"📋 Duplicates removed: {before - len(df)}")
print(f"✅ Remaining: {len(df)} rows")

print("\n" + "=" * 70)
print("STEP 4: ENGINEER FEATURES")
print("=" * 70)

df_model = df.copy()

# ── 1. Convert Area to numeric Marla ──────────────────────────────────────
# 1 Kanal = 20 Marla
def area_to_marla(area_str):
    if pd.isna(area_str) or area_str == 'Unknown':
        return np.nan
    area_str = str(area_str).strip().lower()
    try:
        if 'kanal' in area_str:
            num = float(area_str.replace('kanal', '').strip())
            return num * 20
        elif 'marla' in area_str:
            num = float(area_str.replace('marla', '').strip())
            return num
        else:
            return np.nan
    except:
        return np.nan

df_model['area_marla'] = df_model['Area'].apply(area_to_marla)
median_area = df_model['area_marla'].median()
df_model['area_marla'] = df_model['area_marla'].fillna(median_area)
print(f"✅ Area converted to Marla (numeric)")
print(f"   Range: {df_model['area_marla'].min():.1f} → {df_model['area_marla'].max():.1f} Marla")
print(f"   Median: {median_area:.1f} Marla")

# ── 2. Property Age from Built Year ───────────────────────────────────────
def extract_year(val):
    if pd.isna(val) or str(val).strip() in ('Unknown', '', 'nan'):
        return np.nan
    try:
        yr = int(float(str(val).strip()))
        if 1950 <= yr <= 2030:
            return yr
        return np.nan
    except:
        return np.nan

df_model['built_year_num'] = df_model['Built Year'].apply(extract_year)
median_year = df_model['built_year_num'].median()
df_model['built_year_num'] = df_model['built_year_num'].fillna(median_year)
df_model['property_age'] = 2025 - df_model['built_year_num']
print(f"✅ Property age engineered (2025 - Built Year)")

# ── 3. Bedrooms & Bathrooms ───────────────────────────────────────────────
df_model['Bedrooms'] = pd.to_numeric(df_model['Bedrooms'], errors='coerce')
df_model['Bathrooms'] = pd.to_numeric(df_model['Bathrooms'], errors='coerce')
df_model['Bedrooms'] = df_model['Bedrooms'].fillna(df_model['Bedrooms'].median())
df_model['Bathrooms'] = df_model['Bathrooms'].fillna(df_model['Bathrooms'].median())
print(f"✅ Bedrooms & Bathrooms filled (median)")

# ── 4. Amenity columns (numeric count or 0) ──────────────────────────────
def amenity_to_num(val):
    if pd.isna(val) or str(val).strip() in ('Unknown', '', 'nan'):
        return 0
    if str(val).strip().lower() == 'yes':
        return 1
    try:
        return int(float(str(val).strip()))
    except:
        return 0

for col in ['Parking', 'Servant Qtrs', 'Store Rooms', 'Kitchens']:
    df_model[col + '_num'] = df_model[col].apply(amenity_to_num)

# Binary room flags (Yes/No columns)
for col in ['Drawing Room', 'Dining Room', 'Study Room', 'Prayer Room',
            'Powder Room', 'Lounge/Sitting']:
    df_model[col + '_flag'] = df_model[col].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )
print(f"✅ Amenity features created (numeric counts + binary flags)")

# ── 5. Location encoding ──────────────────────────────────────────────────
# Group rare locations (< 5 properties) as 'Other' to reduce noise
loc_counts = df_model['Location'].value_counts()
rare_locs = loc_counts[loc_counts < 5].index
df_model['Location_grouped'] = df_model['Location'].apply(
    lambda x: x if x not in rare_locs else 'Other'
)
print(f"✅ Locations: {df_model['Location'].nunique()} unique → "
      f"{df_model['Location_grouped'].nunique()} after grouping rare ones")

encoders = {}
le_loc = LabelEncoder()
df_model['location_encoded'] = le_loc.fit_transform(df_model['Location_grouped'].astype(str))
encoders['Location'] = le_loc
encoders['Location_grouped_classes'] = df_model['Location_grouped'].unique().tolist()

le_type = LabelEncoder()
df_model['Property Type'] = df_model['Property Type'].fillna('House')
df_model['type_encoded'] = le_type.fit_transform(df_model['Property Type'].astype(str))
encoders['Property Type'] = le_type

# Also save area_to_marla function info
encoders['area_to_marla'] = 'kanal*20, marla=marla'
encoders['median_area_marla'] = median_area
encoders['median_year'] = median_year

print("\n" + "=" * 70)
print("STEP 5: BUILD FEATURE MATRIX")
print("=" * 70)

final_features = [
    'area_marla',          # KEY: actual numeric area (was label-encoded before)
    'Bedrooms',
    'Bathrooms',
    'location_encoded',
    'type_encoded',
    'property_age',        # NEW: age of property
    'Parking_num',
    'Servant Qtrs_num',
    'Store Rooms_num',
    'Kitchens_num',
    'Drawing Room_flag',   # NEW: binary room flags
    'Dining Room_flag',
    'Study Room_flag',
    'Prayer Room_flag',
    'Lounge/Sitting_flag',
]

X = df_model[final_features].copy()
y = df_model['Price (PKR)'].copy()

# Final NaN check
X = X.fillna(0)

print(f"✅ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"   Features: {final_features}")

print("\n" + "=" * 70)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples")
print(f"💰 Train price — Mean: PKR {y_train.mean():,.0f} | Std: PKR {y_train.std():,.0f}")
print(f"💰 Test  price — Mean: PKR {y_test.mean():,.0f} | Std: PKR {y_test.std():,.0f}")

print("\n" + "=" * 70)
print("STEP 7: SAVE FILES")
print("=" * 70)

df_model.to_csv('preprocessed_data.csv', index=False)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(final_features))

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("   ✔ preprocessed_data.csv")
print("   ✔ X_train.csv / X_test.csv")
print("   ✔ y_train.csv / y_test.csv")
print("   ✔ feature_names.txt")
print("   ✔ encoders.pkl")

print("\n" + "=" * 70)
print("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n✨ Ready for ML — {len(X)} samples × {len(final_features)} features")