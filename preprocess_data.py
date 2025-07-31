import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import os

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def augment_smiles(smiles, num_augmentations=5):
    """SMILES 문자열을 증강하여 여러 개의 동등한 표현을 생성합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * num_augmentations
    
    augmented = set()
    for _ in range(num_augmentations * 10): # 충분한 시도를 통해 다양성 확보
        if len(augmented) >= num_augmentations:
            break
        augmented.add(Chem.MolToSmiles(mol, doRandom=True))
    
    augmented = list(augmented)
    while len(augmented) < num_augmentations:
        augmented.append(smiles) # 목표 개수에 미달 시 원본 추가
        
    return augmented

def preprocess_main():
    """데이터 집계, 가중 샘플링, 데이터 증강을 포함한 전체 전처리 파이프라인입니다."""
    project_root = get_project_root()
    
    # --- 1. 데이터 로드 ---
    print("1. 원본 데이터 로딩...")
    data_path = os.path.join(project_root, 'dataset', 'train', 'train_data_02.csv')
    df = pd.read_csv(data_path)
    print(f"  - 원본 데이터 크기: {len(df)}")

    # --- 2. 데이터 집계 (Data Aggregation) ---
    print("\n2. 데이터 집계 (중복 SMILES 처리)...")
    # 동일 SMILES에 대해 Inhibition 값의 평균 계산
    df_agg = df.groupby('Canonical_Smiles')['Inhibition'].mean().reset_index()
    print(f"  - 집계 후 데이터 크기: {len(df_agg)}")

    # --- 3. 가중 샘플링 (Weighted Sampling) ---
    print("\n3. 가중 샘플링 (소수 데이터 복제)...")
    # Inhibition 값이 극단적인 데이터(0-10, 90-100)를 복제
    extreme_data_low = df_agg[df_agg['Inhibition'] <= 10]
    extreme_data_high = df_agg[df_agg['Inhibition'] >= 90]
    
    # 각 그룹을 2번씩 더 복제 (총 3배)
    df_sampled = pd.concat([df_agg] + [extreme_data_low]*2 + [extreme_data_high]*2, ignore_index=True)
    print(f"  - 가중 샘플링 후 데이터 크기: {len(df_sampled)}")

    # --- 4. 데이터 증강 (Data Augmentation) ---
    print("\n4. 데이터 증강 (SMILES 표현 다양화)...")
    augmented_data = []
    tqdm.pandas(desc="SMILES 증강 진행 중")
    
    # 각 행에 대해 SMILES 증강 적용
    for _, row in df_sampled.progress_apply(lambda x: x, axis=1).iterrows():
        smiles = row['Canonical_Smiles']
        inhibition = row['Inhibition']
        # 각 SMILES를 5개의 다른 표현으로 증강
        augmented_smiles_list = augment_smiles(smiles, num_augmentations=5)
        for aug_smiles in augmented_smiles_list:
            augmented_data.append({'Canonical_Smiles': aug_smiles, 'Inhibition': inhibition})

    df_augmented = pd.DataFrame(augmented_data)
    print(f"  - 데이터 증강 후 최종 데이터 크기: {len(df_augmented)}")

    # --- 5. 최종 데이터셋 저장 ---
    output_path = os.path.join(project_root, 'dataset', 'train', 'train_data_03_final.csv')
    df_augmented.to_csv(output_path, index=False)
    print(f"\n전처리 완료! 최종 데이터셋이 '{output_path}'에 저장되었습니다.")

if __name__ == '__main__':
    preprocess_main() 