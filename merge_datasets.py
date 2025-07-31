import pandas as pd
import os

def merge_and_create_dataset():
    """
    3개의 데이터셋(대회, ChEMBL, BindingDB)을 병합하고 중복을 제거한 후,
    새로운 ID를 부여하여 최종 데이터셋을 생성합니다.
    """
    # --- 1. 파일 경로 정의 ---
    print("1. 파일 경로 설정 중...")
    
    # 입력 파일
    train_csv_path = 'raw_data/data/train.csv'
    chembl_csv_path = 'dataset/cyp3a4_regression_dataset.csv'
    bindingdb_csv_path = 'raw_data/bindingdb_cyp3a4_data.csv'
    
    # 출력 파일
    output_dir = 'dataset'
    output_filename = 'cyp3a4_full_merged_dataset.csv'
    output_path = os.path.join(output_dir, output_filename)

    # --- 2. 데이터셋 로딩 ---
    print("2. 데이터셋 로딩 중...")
    
    input_files = {
        "Contest": train_csv_path,
        "ChEMBL": chembl_csv_path,
        "BindingDB": bindingdb_csv_path
    }
    
    dataframes = []
    for source, path in input_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Inhibition 값이 없는 테스트 데이터 등을 고려하여 컬럼 존재 확인
            if 'Inhibition' in df.columns and 'Canonical_Smiles' in df.columns:
                dataframes.append(df[['Canonical_Smiles', 'Inhibition']])
                print(f" - '{path}' ({source}) 에서 {len(df)}개 레코드 로드 완료.")
            else:
                print(f" - 경고: '{path}'에 필요한 컬럼('Canonical_Smiles', 'Inhibition')이 없어 건너뜁니다.")
        else:
            print(f" - 경고: '{path}' 파일을 찾을 수 없어 건너뜁니다.")

    if not dataframes:
        print("오류: 병합할 데이터가 없습니다. 스크립트를 종료합니다.")
        return

    # --- 3. 데이터프레임 병합 및 정제 ---
    print("\n3. 데이터프레임 병합 및 정제 중...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    # SMILES 기준으로 중복 데이터 제거 (첫 번째 값 유지)
    initial_count = len(combined_df)
    # SMILES가 없는 행 제거
    combined_df.dropna(subset=['Canonical_Smiles'], inplace=True)
    # 중복 제거
    combined_df.drop_duplicates(subset=['Canonical_Smiles'], keep='first', inplace=True)
    final_count = len(combined_df)
    
    print(f" - 총 {initial_count}개 데이터 중 {initial_count - final_count}개의 중복을 제거했습니다.")
    print(f" - 최종 데이터 수: {final_count}개")

    # --- 4. 최종 데이터셋 생성 ---
    print("\n4. 최종 데이터셋 생성 중...")
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.insert(0, 'ID', [f'MERGED_{i+1:05d}' for i in range(final_count)])
    
    # 디렉토리 존재 확인 및 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    combined_df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print("모든 데이터셋 병합 성공!")
    print(f"최종 통합 데이터셋이 '{output_path}'에 저장되었습니다.")
    print(f"총 {final_count}개의 고유 화합물 데이터 확보.")
    print("-" * 50)

if __name__ == '__main__':
    merge_and_create_dataset() 