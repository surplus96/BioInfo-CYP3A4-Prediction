import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from tqdm import tqdm
import subprocess

# RDKit 로깅 비활성화 (불필요한 경고 메시지 숨기기)
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    # 'BioInfo-CYP3A4-Contest'가 나올 때까지 상위 디렉터리로 이동
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

# --- 경로 설정 ---
PROJECT_ROOT = get_project_root()

# 입력 및 출력 경로 설정
TRAIN_INPUT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'train_data.csv')
# 테스트 데이터 경로 수정
TEST_INPUT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'test_data.csv') 
TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'CYP3A4_feature_engineering', 'train_features.csv')
TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'CYP3A4_feature_engineering', 'test_features.csv')

# 피처가 추가된 데이터를 저장할 새로운 디렉토리 생성
# output_dir = 'dataset/train_featured'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

def calculate_descriptors(smiles_string: str) -> pd.Series:
    """
    주어진 SMILES 문자열로부터 RDKit 디스크립터, Morgan Fingerprint, MACCS Keys를 계산합니다.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        # 유효하지 않은 SMILES의 경우, 모든 피처를 0으로 채운 Series 반환
        # 피처 이름은 아래 로직과 일치해야 함
        descriptor_names = [
            'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumHeteroatoms',
            'FractionCSP3', 'NumAliphaticRings', 'HeavyAtomCount',
            'RingCount', 'NOCount', 'NHOHCount'
        ]
        morgan_names = [f'morgan_{i}' for i in range(2048)]
        all_feature_names = descriptor_names + morgan_names
        return pd.Series([0] * len(all_feature_names), index=all_feature_names)

    # 1. 핵심 RDKit 디스크립터 (14개)
    try:
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'RingCount': Descriptors.RingCount(mol),
            'NOCount': Descriptors.NOCount(mol),
            'NHOHCount': Descriptors.NHOHCount(mol)
        }
    except Exception:
        # 예외 발생 시 모든 값을 0으로 채움
        descriptor_names = [
            'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumHeteroatoms',
            'FractionCSP3', 'NumAliphaticRings', 'HeavyAtomCount',
            'RingCount', 'NOCount', 'NHOHCount'
        ]
        morgan_names = [f'morgan_{i}' for i in range(2048)]
        all_feature_names = descriptor_names + morgan_names
        return pd.Series([0] * len(all_feature_names), index=all_feature_names)

    # 2. Morgan Fingerprint (2048 bits)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    morgan_features = {f'morgan_{i}': int(bit) for i, bit in enumerate(morgan_fp.ToBitString())}

    # 모든 피처를 하나의 딕셔너리로 결합
    all_features = {**descriptors, **morgan_features}

    return pd.Series(all_features)


def generate_features(input_path, output_path):
    """
    SMILES 데이터로부터 RDKit 기반 피처들을 생성합니다.
    """
    print(f"데이터 로딩: '{input_path}'")
    df = pd.read_csv(input_path)
    
    # tqdm을 사용하여 진행 상황 표시와 함께 분자 지표 계산
    tqdm.pandas(desc="분자 피처 생성 중")
    features_df = df['Canonical_Smiles'].progress_apply(calculate_descriptors)
    
    # ID와 계산된 피처만 포함하는 새로운 데이터프레임 생성
    result_df = pd.concat([df[['ID', 'Canonical_Smiles']], features_df], axis=1)
    
    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_path, index=False)
    print(f"'{output_path}'에 {features_df.shape[1]}개의 피처가 성공적으로 저장되었습니다.")


if __name__ == '__main__':
    # --- 피처 생성 실행 ---
    # train 데이터에 대해 피처 생성
    print("="*50)
    print("Train 데이터셋 피처 생성 시작")
    generate_features(TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH)
    print("="*50)
    
    print("\n" + "="*50)
    print("Test 데이터셋 피처 생성 시작")
    generate_features(TEST_INPUT_PATH, TEST_OUTPUT_PATH)
    print("="*50) 