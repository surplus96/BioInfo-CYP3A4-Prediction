# CYP3A4_data_miner/main.py

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

from config import (
    TARGET_CHEMBL_ID,
    ACTIVITY_TYPE,
    OUTPUT_DIR,
    OUTPUT_FILENAME,
    RAW_DATA_DIR,
    RAW_DATA_FILENAME,
)

def fetch_cyp3a4_data():
    """
    ChEMBL 데이터베이스에서 CYP3A4 효소에 대한 Inhibition 데이터를 수집합니다.
    """
    print(f"Fetching data for target: {TARGET_CHEMBL_ID}")
    
    activity = new_client.activity
    
    res = activity.filter(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type=ACTIVITY_TYPE
    ).only(
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_type',
        'standard_value',
        'standard_units'
    )
    
    df = pd.DataFrame.from_records(res)
    print(f"Successfully fetched {len(df)} records from ChEMBL.")
    
    # 원본 데이터 저장
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILENAME)
    df.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")
    
    return df

def process_data(df):
    """
    수집된 Inhibition(%) 데이터를 전처리합니다.
    """
    print("Processing raw data for inhibition percentage...")
    
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df.dropna(subset=['molecule_chembl_id', 'canonical_smiles', 'standard_value'], inplace=True)
    df.drop_duplicates(subset=['molecule_chembl_id'], keep='first', inplace=True)
    df = df[df['standard_value'].between(0, 100)]
    
    df.rename(columns={'standard_value': 'Inhibition'}, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    print(f"Data processing complete. {len(df)} compounds remaining.")
    
    return df

def create_final_dataset(df):
    """
    요청된 형식에 맞춰 최종 데이터셋을 생성합니다.
    """
    print("Creating final dataset...")
    
    final_df = df[['canonical_smiles', 'Inhibition']].copy()
    final_df.rename(columns={'canonical_smiles': 'Canonical_Smiles'}, inplace=True)
    
    final_df.insert(0, 'ID', [f"TRAIN_{i+1:04d}" for i in range(len(final_df))])
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print("Dataset creation successful!")
    print(f"Final dataset saved to: {output_path}")
    print(f"Total records in final dataset: {len(final_df)}")
    print("Columns: ", final_df.columns.tolist())
    print("-" * 50)
    
    return final_df

def main():
    """
    전체 데이터 마이닝 파이프라인을 실행합니다.
    """
    print("Starting CYP3A4 Data Miner Pipeline...")
    # Change working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    raw_df = fetch_cyp3a4_data()
    processed_df = process_data(raw_df)
    create_final_dataset(processed_df)
    print("Pipeline finished.")

if __name__ == '__main__':
    main() 