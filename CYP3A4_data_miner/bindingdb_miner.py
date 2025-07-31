# CYP3A4_data_miner/bindingdb_miner.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase

# RDKit 로깅 비활성화
rdBase.DisableLog('rdApp.error')

# --- 설정 ---
INPUT_FILE = "BindingDB_All_2D.sdf"  # 입력 파일 형식 변경
INPUT_DIR = os.path.join("CYP3A4_data_miner", "raw_data") # 경로 수정
OUTPUT_DIR = "raw_data"
OUTPUT_FILENAME = "bindingdb_cyp3a4_data.csv"

# --- 데이터 처리 함수 ---

def convert_ic50_to_pic50(value):
    """IC50 (nM) 값을 pIC50으로 변환합니다."""
    if pd.isna(value) or value <= 0:
        return None
    molar = value * 1e-9
    return -np.log10(molar)

# --- 메인 파이프라인 ---

def main():
    """BindingDB SDF 데이터 처리 파이프라인을 실행합니다."""
    print("--- BindingDB CYP3A4 데이터 마이너 (SDF 버전) 시작 ---")

    input_path = os.path.join(INPUT_DIR, INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다.")
        print("BindingDB 웹사이트에서 'BindingDB_All_2D.sdf' 파일을 다운로드하여")
        print(f"프로젝트 내 '{INPUT_DIR}' 디렉토리에 위치시켜 주세요.")
        return

    # 1. SDF 파일 로드 및 데이터 추출
    print(f"'{input_path}' SDF 파일 처리 중... (시간이 다소 소요될 수 있습니다)")
    supplier = Chem.SDMolSupplier(input_path)
    
    all_records = []
    for mol in tqdm(supplier, desc="SDF 레코드 처리 중"):
        if mol is None:
            continue
        
        try:
            # 타겟 이름과 IC50 값 속성 확인 및 추출
            if "Target Name" in mol.GetPropNames() and "Cytochrome P450 3A4" in mol.GetProp("Target Name"):
                if "IC50 (nM)" in mol.GetPropNames():
                    ic50_str = mol.GetProp("IC50 (nM)")
                    # '>'나 '<' 같은 문자 제거 후 숫자로 변환
                    ic50_val = float(ic50_str.replace('>', '').replace('<', ''))
                    
                    # RDKit mol 객체로부터 직접 Canonical SMILES 생성
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    
                    if smiles and ic50_val > 0:
                        all_records.append({
                            "Canonical_Smiles": smiles,
                            "IC50_nM": ic50_val
                        })
        except (KeyError, ValueError, TypeError):
            # 속성이 없거나 값 변환에 실패하면 건너뛰기
            continue
            
    if not all_records:
        print("CYP3A4 관련 유효 데이터를 찾을 수 없습니다. 스크립트를 종료합니다.")
        return
        
    print(f"필터링 후 {len(all_records)}개의 CYP3A4 관련 레코드 발견.")
    
    # 2. 데이터프레임 변환 및 후처리
    final_df = pd.DataFrame(all_records)
    
    # pIC50 계산
    final_df['pIC50'] = final_df['IC50_nM'].apply(convert_ic50_to_pic50)
    final_df.dropna(subset=['pIC50'], inplace=True)

    # pIC50 값을 0-100 Inhibition 스케일로 변환
    min_pic50, max_pic50 = final_df['pIC50'].min(), final_df['pIC50'].max()
    if max_pic50 > min_pic50:
        final_df['Inhibition'] = 100 * (final_df['pIC50'] - min_pic50) / (max_pic50 - min_pic50)
    else:
        final_df['Inhibition'] = 50.0
        
    # 중복 SMILES 처리 (평균값 사용)
    final_df = final_df.groupby('Canonical_Smiles')['Inhibition'].mean().reset_index()

    # ID 부여
    final_df.insert(0, 'ID', [f'BDB_{i+1:06d}' for i in range(len(final_df))])

    # 3. 최종 파일 저장
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df[['ID', 'Canonical_Smiles', 'Inhibition']].to_csv(output_path, index=False)
    
    print("-" * 50)
    print("BindingDB (SDF) 데이터 마이닝 성공!")
    print(f"최종 데이터가 '{output_path}'에 저장되었습니다.")
    print(f"총 {len(final_df)}개의 화합물 데이터 확보.")
    print("최종 컬럼:", final_df.columns.tolist())
    print("-" * 50)

if __name__ == '__main__':
    main() 