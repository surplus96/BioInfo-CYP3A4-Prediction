import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import optuna
import json
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Optuna 로깅 비활성화
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 대회 평가지표 계산 함수 ---
def normalized_rmse(y_true, y_pred):
    """정규화된 RMSE를 계산합니다."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = np.max(y_true) - np.min(y_true)
    return min(rmse / y_range if y_range != 0 else rmse, 1.0)

def pearson_correlation(y_true, y_pred):
    """피어슨 상관계수를 계산합니다."""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0 # 분산이 0일 경우 상관계수 계산 불가
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return np.clip(corr if not np.isnan(corr) else 0, 0, 1)

def competition_score(y_true, y_pred, y_range=None):
    """대회 평가지표를 계산합니다."""
    nrmse = normalized_rmse(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - nrmse) + 0.5 * pearson

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    # 'BioInfo-CYP3A4-Contest'가 나올 때까지 상위 디렉터리로 이동
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

def generate_rdkit_features(project_root):
    """RDKit 피처 생성 스크립트를 실행합니다."""
    print("RDKit 피처 파일이 없어 새로 생성합니다...")
    script_path = os.path.join(project_root, 'CYP3A4_feature_engineering', 'generate_features.py')
    try:
        subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print("RDKit 피처 생성 완료.")
    except subprocess.CalledProcessError as e:
        print("RDKit 피처 생성 중 오류 발생:")
        print(e.stderr)
        raise

def get_data(project_root):
    """
    모델 학습을 위한 데이터를 준비합니다.
    - generate_features.py에서 생성된 RDKit 피처를 직접 로드합니다.
    - 파일이 없으면 generate_features.py를 실행합니다.
    """
    print("피처 데이터 로딩을 시작합니다...")
    
    feature_dir = os.path.join(project_root, 'CYP3A4_feature_engineering')
    labels_dir = os.path.join(project_root, 'dataset', 'train')

    # 1. 피처 파일 경로 설정
    train_path = os.path.join(feature_dir, 'train_features.csv')
    test_path = os.path.join(feature_dir, 'test_features.csv')

    # 피처 파일이 없으면 생성
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        generate_rdkit_features(project_root)

    print("피처 파일 로딩 중...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 원본 학습 데이터에서 타겟 값 로드
    labels_df = pd.read_csv(os.path.join(labels_dir, 'train_data.csv'))

    # 2. 피처 및 타겟 분리
    # 피처 데이터와 타겟 데이터를 'ID' 기준으로 병합하여 순서 보장
    merged_df = pd.merge(train_df, labels_df[['ID', 'Inhibition']], on='ID')
    
    y = merged_df['Inhibition']
    X = merged_df.drop(columns=['ID', 'Canonical_Smiles', 'Inhibition'])
    
    # 테스트 데이터에서도 동일하게 ID와 SMILES 컬럼 제거
    X_test = test_df.drop(columns=['ID', 'Canonical_Smiles'])
    test_ids = test_df['ID']
    
    # 컬럼 순서 정렬
    X_test = X_test[X.columns]
    
    print(f"데이터 준비 완료. 학습 데이터: {X.shape}, 테스트 데이터: {X_test.shape}")

    return X, y, X_test, test_ids


def screen_and_get_models():
    """
    사전 정의된 고성능 하이퍼파라미터로 초기화된 모델들을 반환합니다.
    """
    print("사전 정의된 하이퍼파라미터로 모델을 초기화합니다...")
    models = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, gamma=0,
            reg_alpha=0.1, reg_lambda=1, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    }
    return models


def train_and_evaluate_tree_models():
    """
    하이퍼파라미터 최적화, 모델 학습, 평가 및 예측을 수행합니다.
    """
    # --- 0. 설정 ---
    print("스크립트 설정...")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    
    # --- 1. 데이터 로딩 ---
    print("데이터 로딩 중...")
    project_root = get_project_root()
    X, y, X_test, test_ids = get_data(project_root)
    
    # 데이터 스케일링 및 결측치 처리
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = np.nan_to_num(X_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    # --- 2. 모델 가져오기 ---
    models = screen_and_get_models()
    models_to_optimize = ["XGBoost", "LightGBM"]

    # --- 3. 하이퍼파라미터 최적화 (Optuna) ---
    print("\n--- 하이퍼파라미터 최적화 시작 ---")

    def objective(trial, model_name, X_data, y_data):
        """Optuna가 호출할 목적 함수"""
        if model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42, 'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42, 'n_jobs': -1
            }
            model = lgb.LGBMRegressor(**params)
        else:
            return 0.0

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = [competition_score(y_data.iloc[val_idx], model.fit(X_data[train_idx], y_data.iloc[train_idx]).predict(X_data[val_idx])) for train_idx, val_idx in kf.split(X_data)]
        return np.mean(scores)

    for name in models_to_optimize:
        print(f"\n'{name}' 모델 최적화 중 (n_trials=50)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, name, X_scaled, y), n_trials=50, show_progress_bar=True)
        
        print(f"'{name}' 최적화 완료. Best CV Score: {study.best_value:.4f}")
        print("최적 하이퍼파라미터:", study.best_params)
        models[name].set_params(**study.best_params)

    # --- 4. 최종 모델 성능 평가 (최적화된 파라미터 적용 후) ---
    print("\n--- 최적화된 모델 5-Fold 교차 검증 시작 ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    performance_results = []
    model_cv_predictions = {}
    
    for name, model in models.items():
        print(f"'{name}' 모델 평가 중...")
        val_preds = np.zeros(len(y))
        for train_index, val_index in tqdm(kf.split(X_scaled), total=kf.get_n_splits(), desc=f"{name} CV"):
            model_clone = joblib.load(joblib.dump(model, 'temp_model.joblib')[0])
            model_clone.fit(X_scaled[train_index], y.iloc[train_index])
            val_preds[val_index] = model_clone.predict(X_scaled[val_index])
        os.remove('temp_model.joblib')

        model_cv_predictions[name] = val_preds
        performance_results.append({
            "Model": name,
            "Competition Score": competition_score(y, val_preds),
            "Normalized RMSE": normalized_rmse(y, val_preds),
            "Pearson Correlation": pearson_correlation(y, val_preds)
        })
        print(f"'{name}' 모델 평가 완료. 대회 점수: {performance_results[-1]['Competition Score']:.4f}")

    performance_df = pd.DataFrame(performance_results).sort_values(by="Competition Score", ascending=False).reset_index(drop=True)
    print("\n--- 모델 성능 요약 (최적화 후) ---")
    print(performance_df.to_string())
    performance_df.to_csv('model_performance_summary_optimized.csv', index=False)
    print("\n최적화된 모델 성능 요약표를 'model_performance_summary_optimized.csv' 파일로 저장했습니다.")

    # --- 5. 모델 학습 및 최종 예측 생성 (가중 평균 앙상블) ---
    print("\n--- 최종 모델 학습 및 가중 평균 앙상블 예측 생성 ---")
    
    # GNN 예측 결과 로드
    gnn_preds_path = os.path.join(project_root, 'CYP3A4_model_train', 'gnn_predictions.csv')
    try:
        gnn_preds_df = pd.read_csv(gnn_preds_path)
        # GNN 예측값을 Inhibition 컬럼 기준으로 추출
        gnn_predictions = gnn_preds_df['Inhibition'].values
        print("GNN 예측 결과를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: GNN 예측 결과 파일 '{gnn_preds_path}'를 찾을 수 없습니다. predict.py를 먼저 실행해야 합니다.")
        # GNN 예측이 없으면 스크립트를 중단하거나, 다른 방식으로 처리할 수 있음
        # 여기서는 GNN 없이 진행하도록 0으로 채움 (실제로는 오류 발생시키는 것이 나을 수 있음)
        gnn_predictions = np.zeros(X_test_scaled.shape[0])


    # 각 모델의 성능(점수)을 가중치로 사용
    weights = performance_df.set_index('Model')['Competition Score']
    
    # GNN 모델에 가중치 부여 (여기서는 트리 모델 중 최고 점수를 부여)
    if 'XGBoost' in weights and 'LightGBM' in weights:
         # 성능이 높은 모델에 더 높은 가중치를 부여하기 위해 제곱을 사용
        weights['GNN'] = max(weights['XGBoost'], weights['LightGBM'])
    else:
        weights['GNN'] = weights.max()

    # 가중치의 합이 1이 되도록 정규화
    total_weight = weights.sum() + weights['GNN'] # GNN 가중치를 포함하여 전체 합 계산
    weights = weights / total_weight
    gnn_weight = weights.pop('GNN') # GNN 가중치 분리

    print("\n적용될 모델별 가중치 (정규화 후):")
    print(weights)
    print(f"GNN 가중치: {gnn_weight:.4f}")

    final_predictions = np.zeros(X_test_scaled.shape[0])
    
    # 트리 모델 예측 및 가중치 적용
    for name, model in models.items():
        print(f"--- '{name}' 모델 전체 데이터 학습 및 예측 ---")
        model.fit(X_scaled, y)
        preds = model.predict(X_test_scaled)
        final_predictions += preds * weights.get(name, 0) # 해당 모델의 가중치를 곱함
        print(f"'{name}' 모델 예측 완료 및 가중치 적용.")
        
    # GNN 예측에 가중치 적용
    final_predictions += gnn_predictions * gnn_weight
    print("GNN 예측에 가중치를 적용했습니다.")
    
    # --- 6. 최종 제출 파일 생성 ---
    submission_df = pd.DataFrame({'ID': test_ids, 'Inhibition': final_predictions})
    submission_df.to_csv('submission_final_ensembled.csv', index=False)
    print("\n최종 앙상블 모델 예측이 완료되었습니다.")
    print("제출 파일: 'submission_final_ensembled.csv'")

    # --- 7. 결과 시각화 (가장 성능이 좋은 모델 기준) ---
    best_model_name = performance_df.loc[0, 'Model']
    best_model_instance = models[best_model_name]
    best_model_cv_preds = model_cv_predictions[best_model_name]
    best_model_score = performance_df.loc[0, "Competition Score"]
    
    print(f"\n--- 결과 시각화 시작 (Best Model: {best_model_name}) ---")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, best_model_cv_preds, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.xlabel('실제값 (Inhibition)')
    plt.ylabel('예측값 (Inhibition)')
    plt.title(f'{best_model_name} 모델 CV 예측 성능 (대회 점수: {best_model_score:.4f})')
    plt.grid(True)
    plt.savefig('model_performance_optimized.png')
    print("모델 성능 시각화 저장 완료: 'model_performance_optimized.png'")

    if hasattr(best_model_instance, 'feature_importances_'):
        n_features = 20
        importances = best_model_instance.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        feature_names = X.columns[indices]
        plt.figure(figsize=(12, 8))
        plt.title(f'{best_model_name} 모델 상위 {n_features}개 특성 중요도 (최적화 후)')
        plt.barh(range(n_features), importances[indices], align='center')
        plt.yticks(range(n_features), feature_names)
        plt.xlabel('특성 중요도')
        plt.tight_layout()
        plt.savefig('feature_importance_optimized.png')
        print("특성 중요도 시각화 저장 완료: 'feature_importance_optimized.png'")


if __name__ == '__main__':
    train_and_evaluate_tree_models() 