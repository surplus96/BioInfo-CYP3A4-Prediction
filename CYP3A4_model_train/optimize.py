import subprocess
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import random

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def run_optimization(n_trials=50):
    """
    Optuna 없이 Random Search를 직접 구현하여 하이퍼파라미터 최적화를 수행합니다.
    """
    project_root = get_project_root()
    split_dir = os.path.join(project_root, 'CYP3A4_model_train', 'temp_split_for_opt')

    best_rmse = float('inf')
    best_params = {}

    try:
        # --- 데이터 준비 (단 한 번만 실행) ---
        print("--- 최적화를 위해 데이터를 미리 분할합니다 ---")
        # 더 작은 데이터셋을 사용하도록 경로 수정
        dataset_path = os.path.join(project_root, 'dataset', 'train', 'train_data_02.csv')
        features_path = os.path.join(project_root, 'dataset', 'train_featured', 'train_features_3d.csv')
        
        print(f"메인 데이터: {dataset_path}")
        print(f"특성 데이터: {features_path}")

        main_df = pd.read_csv(dataset_path)
        features_df = pd.read_csv(features_path, header=None)

        if len(main_df) + 1 != len(features_df):
             print(f"!!! 경고: 메인 데이터(헤더 포함, {len(main_df)+1}줄)와 특성 데이터({len(features_df)}줄)의 길이가 다릅니다!")
             # chemprop 버그를 감안하여 1 차이는 허용
             if abs((len(main_df)+1) - len(features_df)) > 1:
                return

        indices = list(range(len(main_df)))
        train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_df = main_df.iloc[train_idx]
        val_df = main_df.iloc[val_idx]
        train_features_df = features_df.iloc[train_idx]
        val_features_df = features_df.iloc[val_idx]

        os.makedirs(split_dir, exist_ok=True)
        
        train_path = os.path.join(split_dir, 'train.csv')
        val_path = os.path.join(split_dir, 'val.csv')
        train_features_path = os.path.join(split_dir, 'train_features.csv')
        val_features_path = os.path.join(split_dir, 'val_features.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        np.savetxt(train_features_path, train_features_df.values, delimiter=',')
        np.savetxt(val_features_path, val_features_df.values, delimiter=',')

        dummy_row = ','.join(['0'] * train_features_df.shape[1]) + '\n'
        with open(train_features_path, 'a') as f: f.write(dummy_row)
        with open(val_features_path, 'a') as f: f.write(dummy_row)
            
        print(f"--- 데이터 준비 완료. GNN 하이퍼파라미터 최적화를 시작합니다 (n_trials={n_trials}) ---")

        # --- 최적화 루프 ---
        for i in range(n_trials):
            print(f"\n--- Trial {i+1}/{n_trials} 시작 ---")
            
            # 1. 하이퍼파라미터 무작위 생성
            params = {
                'depth': random.choice([3, 4]),
                'hidden_size': random.choice([300, 400, 500, 600]),
                'dropout': random.uniform(0.1, 0.3),
                'ffn_num_layers': random.choice([2, 3]),
                'learning_rate': 10**random.uniform(-3.3, -2.7) # 5e-4 to 2e-3
            }
            print(f"Parameters: {params}")

            # 2. 모델 결과 저장을 위한 임시 디렉터리 생성
            save_dir = os.path.join(project_root, 'CYP3A4_model_train', f'chemprop_trial_{i}')

            # 3. chemprop 명령어 생성
            command = [
                'chemprop_train', '--dataset_type', 'regression',
                '--data_path', train_path, '--separate_val_path', val_path,
                '--features_path', train_features_path, '--separate_val_features_path', val_features_path,
                '--smiles_columns', 'Canonical_Smiles', '--target_columns', 'Inhibition',
                '--save_dir', save_dir, '--epochs', '100', # '--quiet' 제거
                '--depth', str(params['depth']),
                '--hidden_size', str(params['hidden_size']),
                '--dropout', f"{params['dropout']:.4f}",
                '--ffn_num_layers', str(params['ffn_num_layers']),
                '--init_lr', f"{params['learning_rate']:.6f}",
                '--max_lr', f"{params['learning_rate']:.6f}",
            ]
            
            # 4. 학습 실행
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

            # 5. 결과 평가
            current_rmse = float('inf')

            if result.returncode == 0:
                # 학습은 성공, 이제 stdout에서 직접 점수를 파싱
                output_lines = result.stdout.split('\n')
                # 'Overall test rmse' -> 'Validation rmse'로 검색 문자열 변경
                rmse_line = [line for line in output_lines if 'Validation rmse' in line]
                
                if rmse_line:
                    try:
                        # line: 'Validation rmse = 1.1513'
                        parts = rmse_line[0].split()
                        current_rmse = float(parts[3])
                        print(f"Trial {i+1} 완료. RMSE: {current_rmse:.4f}")
                        if current_rmse < best_rmse:
                            best_rmse = current_rmse
                            best_params = params
                            print(f"*** 새로운 최적 RMSE 발견: {best_rmse:.4f} ***")
                    except (IndexError, ValueError):
                        print(f"Warning: Trial {i+1} succeeded, but failed to parse RMSE from output. Stdout below:")
                        print(result.stdout)
                else:
                    # 학습은 성공했는데 결과 라인을 못찾는 경우
                    print(f"Warning: Trial {i+1} succeeded, but no RMSE score found in output. Stdout below:")
                    print(result.stdout)
            else:
                # 학습 자체가 실패한 경우
                print(f"Error: Trial {i+1} failed. Return code: {result.returncode}")
                print("---------- Chemprop Stderr ----------")
                print(result.stderr)
                print("-----------------------------------")


            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)

        # --- 최종 결과 처리 ---
        print("\n\n--- 최적화 완료 ---")
        if best_params:
            print(f"최적의 검증(Validation) RMSE: {best_rmse:.4f}")
            print("최고의 성능을 보인 하이퍼파라미터:")
            print(best_params)

            save_path = os.path.join(project_root, 'CYP3A4_model_train', 'best_gnn_params.json')
            with open(save_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"\n최적의 파라미터가 '{save_path}' 파일에 저장되었습니다.")
        else:
            print("오류: 모든 Trial이 실패하여 최적의 파라미터를 찾지 못했습니다.")

    finally:
        # --- 정리 ---
        print("\n--- 임시 분할 데이터 정리 ---")
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

if __name__ == '__main__':
    run_optimization(n_trials=50) 