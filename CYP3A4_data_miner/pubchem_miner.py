# CYP3A4_data_miner/pubchem_miner.py

import os
import time
import pandas as pd
import numpy as np
import requests
from io import StringIO
from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase

# RDKit 로깅 비활성화
rdBase.DisableLog('rdApp.error')

# --- 설정 ---
TARGET_UNIPROT_ACCESSION = "P08684"  # Human CYP3A4
OUTPUT_DIR = "raw_data"
OUTPUT_FILENAME = "pubchem_cyp3a4_data.csv"
MIN_RECORDS_PER_ASSAY = 10 # 최소 레코드 수를 만족하지 않는 Assay는 건너뛰기

# --- API 헬퍼 함수 ---

def find_cyp3a4_assays(accession):
    """PubChem에서 특정 단백질 Accession ID와 관련된 Assay ID 목록을 검색합니다."""
    print(f"UniProt Accession '{accession}'에 대한 PubChem BioAssay 검색 중...")
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    search_url = f"{base_url}/assay/target/accession/{accession}/aids/json"
    
    try:
        response = requests.get(search_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        aids = data.get('IdentifierList', {}).get('AID', [])
        
        if not aids:
            print("  - 관련 Assay ID를 찾을 수 없습니다.")
        else:
            print(f"총 {len(aids)}개의 관련 Assay ID를 찾았습니다.")
        return aids
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return []

def fetch_assay_data(aid):
    """주어진 AID에 대한 데이터를 CSV로 받아 데이터프레임으로 반환합니다."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/csv"
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200 and response.content:
            return pd.read_csv(StringIO(response.text))
    except (requests.exceptions.RequestException, pd.errors.ParserError) as e:
        print(f"  - AID {aid} 처리 중 오류: {e}")
    return None

# --- 데이터 처리 함수 ---

def standardize_smiles(smiles):
    """SMILES를 표준화된 형식(Canonical)으로 변환합니다."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) if mol else None
    except:
        return None

def convert_ic50_to_pic50(value, unit):
    """IC50 값을 pIC50으로 변환합니다."""
    unit = str(unit).lower()
    if unit == 'nm':
        molar = value * 1e-9
    elif unit == 'um' or unit == 'µm': # 마이크로몰
        molar = value * 1e-6
    else:
        return None # 지원하지 않는 단위
        
    if molar > 0:
        return -np.log10(molar)
    return None

# --- 메인 파이프라인 ---

def main():
    """PubChem 데이터 수집 및 처리 파이프라인을 실행합니다."""
    print("--- PubChem CYP3A4 데이터 마이너 시작 ---")
    
    # 1. Assay ID 수집
    aids = find_cyp3a4_assays(TARGET_UNIPROT_ACCESSION)
    if not aids:
        print("데이터를 수집할 Assay가 없습니다. 스크립트를 종료합니다.")
        return

    # 2. 데이터 수집 및 정제
    all_processed_data = []
    
    for aid in tqdm(aids, desc="Assay 데이터 처리 중"):
        df = fetch_assay_data(aid)
        if df is None or len(df) < MIN_RECORDS_PER_ASSAY:
            continue

        # 필수 컬럼 확인
        required_cols = ['PUBCHEM_CID', 'PUBCHEM_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME']
        if not all(col in df.columns for col in required_cols):
            continue
            
        # 활성 데이터 컬럼 찾기 (IC50, % inhibition 등)
        activity_col, unit = None, None
        if 'Activity' in df.columns and any(df['Activity'].dropna().astype(str).str.contains('%')):
             activity_col, unit = 'Activity', '%'
        elif 'IC50' in df.columns and 'Unit' in df.columns:
             activity_col, unit = 'IC50', df['Unit'].iloc[0]

        if activity_col is None:
            continue
            
        df_processed = df[['PUBCHEM_SMILES', activity_col]].copy()
        df_processed.columns = ['SMILES', 'Value']
        df_processed['Unit'] = unit

        df_processed.dropna(inplace=True)
        df_processed['Value'] = pd.to_numeric(df_processed['Value'], errors='coerce')
        df_processed.dropna(subset=['Value'], inplace=True)
        
        all_processed_data.append(df_processed)

    if not all_processed_data:
        print("처리할 데이터가 없습니다. 스크립트를 종료합니다.")
        return

    # 3. 데이터 통합 및 표준화
    print("\n수집된 모든 데이터 통합 및 최종 처리 중...")
    combined_df = pd.concat(all_processed_data, ignore_index=True)

    # SMILES 표준화
    combined_df['Canonical_Smiles'] = combined_df['SMILES'].progress_apply(standardize_smiles)
    combined_df.dropna(subset=['Canonical_Smiles'], inplace=True)

    # 활성 값을 pIC50 또는 Inhibition(%)으로 변환
    is_percent_data = combined_df['Unit'] == '%'
    df_percent = combined_df[is_percent_data].copy()
    df_ic50 = combined_df[~is_percent_data].copy()

    final_data = []

    if not df_percent.empty:
        df_percent.rename(columns={'Value': 'Inhibition'}, inplace=True)
        final_data.append(df_percent[['Canonical_Smiles', 'Inhibition']])

    if not df_ic50.empty:
        df_ic50['pIC50'] = df_ic50.apply(lambda row: convert_ic50_to_pic50(row['Value'], row['Unit']), axis=1)
        df_ic50.dropna(subset=['pIC50'], inplace=True)

        # pIC50을 0-100 스케일의 Inhibition 값으로 변환
        min_pic50, max_pic50 = df_ic50['pIC50'].min(), df_ic50['pIC50'].max()
        if max_pic50 > min_pic50:
            df_ic50['Inhibition'] = 100 * (df_ic50['pIC50'] - min_pic50) / (max_pic50 - min_pic50)
            final_data.append(df_ic50[['Canonical_Smiles', 'Inhibition']])
    
    if not final_data:
        print("최종 데이터 생성에 실패했습니다.")
        return

    final_df = pd.concat(final_data, ignore_index=True)
    
    # 중복 SMILES 처리 (평균값 사용)
    final_df = final_df.groupby('Canonical_Smiles')['Inhibition'].mean().reset_index()

    # ID 부여
    final_df.insert(0, 'ID', [f'PUBCHEM_{i+1:05d}' for i in range(len(final_df))])

    # 4. 파일 저장
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print("PubChem 데이터 마이닝 성공!")
    print(f"최종 데이터가 '{output_path}'에 저장되었습니다.")
    print(f"총 {len(final_df)}개의 화합물 데이터 확보.")
    print("최종 컬럼:", final_df.columns.tolist())
    print("-" * 50)

if __name__ == '__main__':
    main() 