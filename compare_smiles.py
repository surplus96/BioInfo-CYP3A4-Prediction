import pandas as pd
import numpy as np
import os

def compare_datasets():
    """
    두 데이터셋을 비교하여 공통된 SMILES와 Inhibition 값의 일치 여부를 확인합니다.
    """
    # 프로젝트 루트를 기준으로 한 상대 경로 (수정된 경로)
    train_csv_path = 'raw_data/data/train.csv'
    generated_csv_path = 'dataset/cyp3a4_regression_dataset.csv'

    # 파일 존재 여부 확인
    if not os.path.exists(train_csv_path):
        print(f"오류: '{train_csv_path}' 파일을 찾을 수 없습니다. 현재 작업 디렉토리: {os.getcwd()}")
        return
    if not os.path.exists(generated_csv_path):
        print(f"오류: '{generated_csv_path}' 파일을 찾을 수 없습니다. 먼저 main.py를 실행하여 파일을 생성해야 합니다.")
        return

    print(f"1. '{train_csv_path}' 파일 로딩 중...")
    # train.csv의 헤더를 자동으로 인식하도록 수정
    train_df = pd.read_csv(train_csv_path)
    # 컬럼 이름 통일
    train_df.rename(columns={'Canonical_Smiles': 'SMILES', 'Inhibition': 'Inhibition_train'}, inplace=True)

    print(f"2. '{generated_csv_path}' 파일 로딩 중...")
    generated_df = pd.read_csv(generated_csv_path)
    generated_df.rename(columns={'Canonical_Smiles': 'SMILES', 'Inhibition': 'Inhibition_generated'}, inplace=True)

    print("\n3. SMILES 값을 기준으로 데이터 병합 및 비교 중...")
    merged_df = pd.merge(train_df, generated_df[['SMILES', 'Inhibition_generated']], on='SMILES', how='inner')

    common_smiles_count = len(merged_df)

    print("-" * 50)
    print("비교 결과:")
    if common_smiles_count == 0:
        print("두 파일 간에 동일한 SMILES 값이 존재하지 않습니다.")
        return

    print(f"총 {common_smiles_count}개의 동일한 SMILES 값을 찾았습니다.")

    # Inhibition 값 비교
    merged_df['values_match'] = np.isclose(merged_df['Inhibition_train'], merged_df['Inhibition_generated'])
    mismatched_df = merged_df[~merged_df['values_match']]

    if mismatched_df.empty:
        print("모든 공통 SMILES에 대해 Inhibition 값이 정확히 일치합니다.")
    else:
        print(f"\n경고: {len(mismatched_df)}개의 SMILES에서 Inhibition 값이 일치하지 않습니다.")
        print("=" * 60)
        for index, row in mismatched_df.head().iterrows():
            print(f"SMILES: {row['SMILES']}")
            print(f"  - train.csv 값: {row['Inhibition_train']}")
            print(f"  - 생성된 파일 값:  {row['Inhibition_generated']}\n")
        if len(mismatched_df) > 5:
            print(f"... 외 {len(mismatched_df) - 5}개의 불일치 항목이 더 있습니다.")
        print("=" * 60)

if __name__ == '__main__':
    compare_datasets() 