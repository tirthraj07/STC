import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt

'''
###################################################################
    DATASET CREATION PHASE
    
    Objectives:
        1. Create a global testing dataset
        2. Create a training and testing dataset for each client
        3. Make it uneven to resemble real world scenario
        4. Plot the graphs in respective folders
###################################################################
'''

def plot_distributions(data, folder_path):
    plt.figure(figsize=(15, 12))
    
    # Age distribution (years)
    plt.subplot(4, 3, 1)
    plt.hist(data['age_years'], bins=20, color='blue', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')
    
    # Height distribution (cm)
    plt.subplot(4, 3, 2)
    plt.hist(data['height'], bins=20, color='green', alpha=0.7)
    plt.title('Height Distribution')
    plt.xlabel('Height (cm)')
    
    # Weight distribution (kg)
    plt.subplot(4, 3, 3)
    plt.hist(data['weight'], bins=20, color='purple', alpha=0.7)
    plt.title('Weight Distribution')
    plt.xlabel('Weight (kg)')
    
    # Gender distribution
    plt.subplot(4, 3, 4)
    plt.bar(data['gender'].value_counts().index, data['gender'].value_counts().values, color='orange', alpha=0.7)
    plt.title('Gender Distribution')
    plt.xlabel('Gender (1=Female, 2=Male)')
    
    # Systolic blood pressure (ap_hi) distribution
    plt.subplot(4, 3, 5)
    plt.hist(data['ap_hi'], bins=20, color='cyan', alpha=0.7)
    plt.title('Systolic Blood Pressure Distribution')
    plt.xlabel('Systolic BP (ap_hi)')
    
    # Diastolic blood pressure (ap_lo) distribution
    plt.subplot(4, 3, 6)
    plt.hist(data['ap_lo'], bins=20, color='pink', alpha=0.7)
    plt.title('Diastolic Blood Pressure Distribution')
    plt.xlabel('Diastolic BP (ap_lo)')
    
    # Cholesterol distribution
    plt.subplot(4, 3, 7)
    plt.bar(data['cholesterol'].value_counts().index, data['cholesterol'].value_counts().values, color='red', alpha=0.7)
    plt.title('Cholesterol Distribution')
    plt.xlabel('Cholesterol Levels (1=Normal, 2=Above, 3=Well Above)')
    
    # Glucose levels distribution
    plt.subplot(4, 3, 8)
    plt.bar(data['gluc'].value_counts().index, data['gluc'].value_counts().values, color='magenta', alpha=0.7)
    plt.title('Glucose Levels Distribution')
    plt.xlabel('Glucose Levels (1=Normal, 2=Above, 3=Well Above)')
    
    # Smoking (binary) distribution
    plt.subplot(4, 3, 9)
    plt.bar(data['smoke'].value_counts().index, data['smoke'].value_counts().values, color='yellow', alpha=0.7)
    plt.title('Smoking Distribution')
    plt.xlabel('Smoking (0=No, 1=Yes)')
    
    # Alcohol intake (binary) distribution
    plt.subplot(4, 3, 10)
    plt.bar(data['alco'].value_counts().index, data['alco'].value_counts().values, color='brown', alpha=0.7)
    plt.title('Alcohol Intake Distribution')
    plt.xlabel('Alcohol (0=No, 1=Yes)')
    
    # Physical activity (binary) distribution
    plt.subplot(4, 3, 11)
    plt.bar(data['active'].value_counts().index, data['active'].value_counts().values, color='teal', alpha=0.7)
    plt.title('Physical Activity Distribution')
    plt.xlabel('Physical Activity (0=No, 1=Yes)')
    
    # Cardiovascular disease (cardio) distribution
    plt.subplot(4, 3, 12)
    plt.bar(data['cardio'].value_counts().index, data['cardio'].value_counts().values, color='brown', alpha=0.7)
    plt.title('Cardiovascular Disease Distribution')
    plt.xlabel('Cardio (0=Absence, 1=Presence)')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{folder_path}/distributions.png')
    plt.close()


def split_save_and_plot(client_data, client_name):
    # Split data into 80% training and 20% testing
    train, test = train_test_split(client_data, test_size=0.2, random_state=42)
    
    # Save training and testing data
    train.to_csv(f'./dataset/{client_name}/training/train.csv', index=False)
    test.to_csv(f'./dataset/{client_name}/testing/test.csv', index=False)
    
    print(client_name)
    print(f"Training : {len(train)}")
    print(f"Testing : {len(test)}")

    # Plot and save distributions for training data
    plot_distributions(train, f'./dataset/{client_name}/training/plots')
    
    # Plot and save distributions for testing data
    plot_distributions(test, f'./dataset/{client_name}/testing/plots')

clients = ['client1', 'client2', 'client3']
folders = ['training', 'testing']
for client in clients:
    for folder in folders:
        os.makedirs(f'./dataset/{client}/{folder}/plots', exist_ok=True)

os.makedirs(f'./dataset/global/plots', exist_ok=True)

dataset = pd.read_csv('./dataset/cleaned_health_data.csv')
dataset['age_years'] = dataset['age'] / 365

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
global_train, global_test = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['cardio'])

global_test.to_csv('./dataset/global/test.csv', index=False)
plot_distributions(global_test,'./dataset/global/plots')

client1_data, client2_data, client3_data = np.array_split(global_train, 3)

# cardio_0 = global_train[global_train['cardio'] == 0]  # cardio = 0 (absence of disease)
# cardio_1 = global_train[global_train['cardio'] == 1]  # cardio = 1 (presence of disease)


# # Client 1: Skewed towards cardio = 0 (80% cardio = 0, 20% cardio = 1)
# client1_cardio_0 = cardio_0.sample(frac=0.8, random_state=42)
# client1_cardio_1 = cardio_1.sample(frac=0.2, random_state=42)
# client1_data = pd.concat([client1_cardio_0, client1_cardio_1]).sample(frac=1, random_state=42)

# # Client 2: Skewed towards cardio = 1 (40% cardio = 0, 60% cardio = 1)
# client2_cardio_0 = cardio_0.drop(client1_cardio_0.index).sample(frac=0.4, random_state=42)
# client2_cardio_1 = cardio_1.drop(client1_cardio_1.index).sample(frac=0.6, random_state=42)
# client2_data = pd.concat([client2_cardio_0, client2_cardio_1]).sample(frac=1, random_state=42)

# # Client 3: Skewed towards cardio = 1 (30% cardio = 0, 70% cardio = 1)
# client3_cardio_0 = cardio_0.drop(client1_cardio_0.index).drop(client2_cardio_0.index)
# client3_cardio_1 = cardio_1.drop(client1_cardio_1.index).drop(client2_cardio_1.index)
# client3_data = pd.concat([client3_cardio_0.sample(frac=0.3, random_state=42), 
#                           client3_cardio_1.sample(frac=0.7, random_state=42)]).sample(frac=1, random_state=42)

clients_data = [client1_data, client2_data, client3_data]

# Iterate over each client's data
for i, data in enumerate(clients_data):
    split_save_and_plot(data, clients[i])
