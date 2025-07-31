import subprocess
import os

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def predict_with_gnn():
    """
    학습된 GNN(Chemprop) 모델로 테스트 데이터에 대한 예측을 수행합니다.
    """
    project_root = get_project_root()
    
    # --- 경로 설정 ---
    # 테스트 데이터는 원본 test.csv를 사용 (SMILES 컬럼 필요)
    test_data_path = os.path.join(project_root, 'raw_data', 'data', 'test.csv')
    # 방금 학습시킨 scaffold 모델 디렉터리
    model_dir = os.path.join(project_root, 'CYP3A4_model_train', 'chemprop_scaffold_model')
    # GNN 예측 결과를 저장할 파일 경로
    prediction_path = os.path.join(project_root, 'CYP3A4_model_train', 'gnn_predictions.csv')

    # --- 필수 파일/디렉터리 존재 여부 확인 ---
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"오류: 테스트 파일 '{test_data_path}'를 찾을 수 없습니다.")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"오류: 학습된 모델 디렉터리 '{model_dir}'을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")

    # --- Chemprop 예측 명령어 생성 ---
    command = [
        'chemprop_predict',
        '--test_path', test_data_path,
        '--checkpoint_dir', model_dir,
        '--preds_path', prediction_path,
        '--smiles_column', 'Canonical_Smiles' # 테스트 데이터의 SMILES 컬럼명
    ]

    print("="*50)
    print("GNN 모델 예측을 시작합니다.")
    print(f"  - 테스트 데이터: {test_data_path}")
    print(f"  - 학습된 모델: {model_dir}")
    print(f"  - 예측 결과 저장: {prediction_path}")
    print("="*50)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"\n성공! GNN 예측 결과가 '{prediction_path}'에 저장되었습니다.")
        print("이제 이 예측 결과를 다른 모델들과 앙상블할 수 있습니다.")

    except subprocess.CalledProcessError as e:
        print(f"\n--- GNN 예측 중 오류 발생 ---")
        print(f"오류 메시지:\n{e.stderr}")
        raise

if __name__ == '__main__':
    predict_with_gnn() 