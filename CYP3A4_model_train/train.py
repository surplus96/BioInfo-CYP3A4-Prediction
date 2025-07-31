import subprocess
import os
import json

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def train_chemprop_model():
    """
    최적화된 하이퍼파라미터와 고급 앙상블 기법을 사용하여
    최종 Chemprop 모델을 학습합니다.
    """
    project_root = get_project_root()
    
    # --- 경로 설정 ---
    dataset_path = os.path.join(project_root, 'dataset', 'train', 'train_data_02.csv')
    params_path = os.path.join(project_root, 'CYP3A4_model_train', 'best_gnn_params.json')
    save_dir = os.path.join(project_root, 'CYP3A4_model_train', 'chemprop_final_model')
    
    os.makedirs(save_dir, exist_ok=True)

    # --- 최적 하이퍼파라미터 로드 ---
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"최적 하이퍼파라미터를 로드했습니다: {params_path}")
    except FileNotFoundError:
        print(f"경고: 최적 파라미터 파일 '{params_path}'를 찾을 수 없습니다. 기본값으로 학습을 진행합니다.")
        best_params = {
            'depth': 3,
            'hidden_size': 300,
            'dropout': 0.1,
            'ffn_num_layers': 2
        }

    print("\n--- 최종 Chemprop 모델 학습 시작 ---")
    print(f"  - 데이터셋: {dataset_path}")
    print(f"  - 저장 디렉터리: {save_dir}")
    print(f"  - 적용 파라미터: {best_params}")

    # --- Chemprop 학습 명령어 생성 ---
    command = [
        'chemprop_train',
        '--data_path', dataset_path,
        '--dataset_type', 'regression',
        '--smiles_columns', 'Canonical_Smiles',
        '--target_columns', 'Inhibition',
        '--save_dir', save_dir,
        '--split_type', 'scaffold_balanced',
        '--quiet',
        
        # 최적화된 하이퍼파라미터 적용
        '--depth', str(best_params.get('depth', 3)),
        '--hidden_size', str(best_params.get('hidden_size', 300)),
        '--dropout', str(best_params.get('dropout', 0.1)),
        '--ffn_num_layers', str(best_params.get('ffn_num_layers', 2)),
        
        # 성능 극대화를 위한 최종 학습 설정
        '--epochs', '100',
        '--num_folds', '5',         # 5-Fold Cross-Validation
        '--ensemble_size', '3',     # 3개 모델 앙상블
        
        # RDKit 피처 보조 활용
        '--features_generator', 'rdkit_2d_normalized',
        '--no_features_scaling'
    ]

    try:
        print("\n최종 모델 학습을 시작합니다. 시간이 매우 오래 걸릴 수 있습니다...")
        subprocess.run(command, check=True)
        print("\n--- 최종 Chemprop 모델 학습 완료 ---")
        print(f"학습된 최종 모델이 '{save_dir}'에 성공적으로 저장되었습니다.")

    except subprocess.CalledProcessError as e:
        print("\n--- 최종 모델 학습 중 오류 발생 ---")
        print(f"오류 메시지: {e}")
    except FileNotFoundError:
        print("\n--- 오류: 'chemprop_train' 명령을 찾을 수 없습니다. ---")
        print("Chemprop이 설치된 conda/virtual 환경이 올바르게 활성화되었는지 확인해주세요.")

if __name__ == '__main__':
    train_chemprop_model() 