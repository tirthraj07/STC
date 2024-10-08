'''
###################################################################
    Performing Data Cleaning
###################################################################
'''
import pandas as pd

# Load original dataset
dataset = pd.read_csv('./dataset/health_data.csv')

# Remove outliers
# Height: keep only values between 100 and 250 cm
dataset_cleaned = dataset[(dataset['height'] >= 100) & (dataset['height'] <= 250)]

# Weight: keep only values between 30 and 150 kg
dataset_cleaned = dataset_cleaned[(dataset_cleaned['weight'] >= 30) & (dataset_cleaned['weight'] <= 150)]

# Systolic Blood Pressure (ap_hi): keep only values between 90 and 200 mmHg
dataset_cleaned = dataset_cleaned[(dataset_cleaned['ap_hi'] >= 90) & (dataset_cleaned['ap_hi'] <= 200)]

# Diastolic Blood Pressure (ap_lo): keep only values between 60 and 120 mmHg
dataset_cleaned = dataset_cleaned[(dataset_cleaned['ap_lo'] >= 60) & (dataset_cleaned['ap_lo'] <= 120)]

# Save the cleaned dataset
dataset_cleaned.to_csv('./dataset/cleaned_health_data.csv', index=False)


print(f"Dataset cleaned. Remaining rows: {len(dataset_cleaned)}")

# Load the new dataset
dataset = pd.read_csv('./dataset/cleaned_health_data.csv')

cols = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']

for col in cols:
    min_col = dataset[col].min()
    max_col = dataset[col].max()
    mean_col = dataset[col].mean() 
    median_col = dataset[col].median()
    mode_col = dataset[col].mode()[0]

    print(f"Minimum {col}: {min_col}")
    print(f"Maximum {col}: {max_col}")
    print(f"Mean {col}: {mean_col}")
    print(f"Median {col}: {median_col}")
    print(f"Mode {col}: {mode_col}")
    print('\n\n')


'''
Before :
Length of Dataset: 70,000

Minimum height: 55
Maximum height: 250
Mean height: 164.35922857142856
Median height: 165.0
Mode height: 165

Minimum weight: 10.0
Maximum weight: 200.0
Mean weight: 74.20569
Median weight: 72.0
Mode weight: 65.0

Minimum ap_hi: -150
Maximum ap_hi: 16020
Mean ap_hi: 128.8172857142857
Median ap_hi: 120.0
Mode ap_hi: 120

Minimum ap_lo: -70
Maximum ap_lo: 11000
Mean ap_lo: 96.63041428571428
Median ap_lo: 80.0
Mode ap_lo: 80

After:
Dataset cleaned. Remaining rows: 68336
Removed : 1664 rows

Minimum height: 100
Maximum height: 250
Mean height: 164.40287696089908
Median height: 165.0
Mode height: 165

Minimum weight: 30.0
Maximum weight: 150.0
Mean weight: 74.06182831889487
Median weight: 72.0
Mode weight: 65.0

Minimum ap_hi: 90
Maximum ap_hi: 200
Mean ap_hi: 126.63856532428002
Median ap_hi: 120.0
Mode ap_hi: 120

Minimum ap_lo: 60
Maximum ap_lo: 120
Mean ap_lo: 81.32002165769141
Median ap_lo: 80.0
Mode ap_lo: 80
'''
