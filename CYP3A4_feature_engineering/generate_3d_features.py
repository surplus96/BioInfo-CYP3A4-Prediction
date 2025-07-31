import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
import numpy as np
from tqdm import tqdm
import os

def generate_3d_features(smiles_string):
    """
    SMILES 문자열로부터 3D 분자 구조를 생성하고 관련 특성을 계산합니다.

    Args:
        smiles_string (str): 분자의 SMILES 표기.

    Returns:
        dict: 계산된 3D 특성들을 담은 딕셔너리. 특성 계산에 실패하면 None을 반환합니다.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        # 3D 특성 계산
        features = {
            'Asphericity': Descriptors3D.Asphericity(mol),
            'Eccentricity': Descriptors3D.Eccentricity(mol),
            'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
            'NPR1': Descriptors3D.NPR1(mol),
            'NPR2': Descriptors3D.NPR2(mol),
            'PMI1': Descriptors3D.PMI1(mol),
            'PMI2': Descriptors3D.PMI2(mol),
            'PMI3': Descriptors3D.PMI3(mol),
            'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
            'SpherocityIndex': Descriptors3D.SpherocityIndex(mol)
        }
        return features
    except Exception as e:
        # print(f"Error processing SMILES {smiles_string}: {e}")
        return None

def process_dataset(input_path, output_path, smiles_col='Canonical_Smiles'):
    """
    주어진 데이터셋을 읽어 각 SMILES에 대한 3D 특성을 계산하고 CSV 파일로 저장합니다.
    Chemprop의 --features_path 형식에 맞게 숫자 특성만, 헤더 없이 저장합니다.
    """
    print(f"{input_path} 파일 처리 중...")
    df = pd.read_csv(input_path)

    tqdm.pandas(desc="SMILES로부터 3D 특성 생성 중")
    features_list = df[smiles_col].progress_apply(generate_3d_features).tolist()

    num_failed = features_list.count(None)

    # 3D 특성 계산에 실패하면 10개의 0으로 구성된 리스트를 사용합니다.
    # 성공하면 딕셔너리의 값들을 리스트로 변환합니다. (Python 3.7+에서 순서 보장)
    default_features = [0.0] * 10
    processed_features = [
        list(f.values()) if f is not None else default_features
        for f in features_list
    ]

    # 리스트의 리스트로부터 데이터프레임을 생성하여 안정성을 높입니다.
    features_df = pd.DataFrame(processed_features)

    # 결과 저장 (Chemprop 호환성을 위해 헤더와 인덱스 제외)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False, header=False)

    print(f"성공적으로 {output_path} 파일에 저장했습니다.")
    print(f"총 {len(df)}개 분자 중 {len(df) - num_failed}개 성공, {num_failed}개는 실패하여 특성을 0으로 채웠습니다.")


if __name__ == "__main__":
    # 최종 학습 데이터(train_data_03_final.csv)에 대한 3D 특성 생성
    train_input = 'dataset/train/train_data_03_final.csv'
    train_output = 'dataset/train_featured/train_features_3d.csv'
    print(f"--- '{train_input}'에 대한 3D 특성 생성 시작 ---")
    process_dataset(train_input, train_output, smiles_col='Canonical_Smiles')

    # 테스트 데이터에 대한 3D 특성 생성 (기존과 동일)
    test_input = 'raw_data/data/test.csv'
    test_output = 'dataset/train_featured/test_features_3d.csv'
    print(f"\n--- '{test_input}'에 대한 3D 특성 생성 시작 ---")
    process_dataset(test_input, test_output, smiles_col='Canonical_Smiles') 