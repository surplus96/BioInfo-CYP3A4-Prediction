
import os
import subprocess
import pandas as pd
from tqdm import tqdm

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    # 'BioInfo-CYP3A4-Contest'가 나올 때까지 상위 디렉터리로 이동
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def run_feature_extraction(project_root, data_type):
    """
    chemprop을 사용하여 GNN 기반 피처를 추출합니다.
    
    :param project_root: 프로젝트 루트 경로
    :param data_type: 'train' 또는 'test'
    """
    print(f"'{data_type}_data.csv'에 대한 GNN 피처 추출을 시작합니다...")

    dataset_path = os.path.join(project_root, 'dataset/train', f'{data_type}_data.csv')
    output_path = os.path.join(project_root, 'CYP3A4_feature_engineering', f'{data_type}_gnn_features.csv')
    model_path = os.path.join(project_root, 'CYP3A4_model_train', 'chemprop_model_hyperopt')

    # chemprop 명령어 생성
    # chemprop_fingerprint 명령어를 사용하여 GNN 기반의 피처 벡터를 추출합니다.
    command = [
        'chemprop_fingerprint',
        '--test_path', dataset_path,
        '--checkpoint_dir', model_path,
        '--preds_path', output_path,
        '--smiles_columns', 'Canonical_Smiles',
    ]

    try:
        print(f"실행 명령어: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        
        # 피처 추출 후 ID 열 추가
        print(f"'{data_type}' 데이터에 ID 열을 추가합니다.")
        gnn_features_df = pd.read_csv(output_path)
        original_data_df = pd.read_csv(dataset_path)

        # ID 열을 GNN 피처 데이터프레임의 첫 번째 열로 삽입
        gnn_features_df.insert(0, 'ID', original_data_df['ID'])

        # chemprop이 예측한 Inhibition 열이 있다면 제거
        if 'Inhibition' in gnn_features_df.columns:
            gnn_features_df = gnn_features_df.drop(columns=['Inhibition'])
            
        gnn_features_df.to_csv(output_path, index=False)
        print(f"ID가 추가된 GNN 피처가 '{output_path}'에 성공적으로 저장되었습니다.")
        return True

    except FileNotFoundError:
        print("오류: 'chemprop'이 설치되어 있지 않거나 경로에 없습니다.")
        print("가상 환경을 활성화하고 'pip install chemprop'을 실행하세요.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"'{data_type}' 데이터 피처 추출 중 오류 발생 (종료 코드: {e.returncode}):")
        print("----- STDERR -----")
        print(e.stderr)
        print("----- STDOUT -----")
        print(e.stdout)
        return False
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    try:
        project_root = get_project_root()
        print(f"프로젝트 루트: {project_root}")

        # 피처를 저장할 디렉터리가 없으면 생성
        feature_dir = os.path.join(project_root, 'CYP3A4_feature_engineering')
        os.makedirs(feature_dir, exist_ok=True)
        
        if run_feature_extraction(project_root, 'train'):
            run_feature_extraction(project_root, 'test')

    except Exception as e:
        print(f"스크립트 실행 중 오류: {e}")

if __name__ == '__main__':
    main() 