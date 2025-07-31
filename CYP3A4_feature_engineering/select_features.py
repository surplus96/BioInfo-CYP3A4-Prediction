import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
import lightgbm as lgb

def get_project_root():
    """현재 스크립트의 위치를 기준으로 프로젝트 루트 디렉터리를 찾습니다."""
    current_path = os.path.abspath(__file__)
    # 'BioInfo-CYP3A4-Contest'가 나올 때까지 상위 디렉터리로 이동
    while os.path.basename(current_path) != 'BioInfo-CYP3A4-Contest':
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            raise Exception("프로젝트 루트 'BioInfo-CYP3A4-Contest'를 찾을 수 없습니다.")
    return current_path

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------


def _remove_low_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """상수(features with zero variance) 제거."""
    selector = VarianceThreshold(threshold=threshold)
    mask = selector.fit(df).get_support()
    removed = df.columns[~mask]
    if len(removed):
        print(f"  - 낮은 분산으로 제거된 피처: {len(removed)}개")
    return df.loc[:, mask]


def _remove_high_nan(df: pd.DataFrame, nan_ratio: float = 0.5) -> pd.DataFrame:
    """결측 비율이 높은 컬럼 제거."""
    ratio = df.isna().mean()
    keep_cols = ratio[ratio < nan_ratio].index
    removed = ratio[ratio >= nan_ratio].index
    if len(removed):
        print(f"  - NaN 비율>{nan_ratio*100:.0f}% 제거 피처: {len(removed)}개")
    return df[keep_cols]


def _correlation_filter(features: pd.DataFrame, target: pd.Series, min_corr: float = 0.05, top_k: int = 800):
    """타깃과 상관이 낮은 피처 제거 후 상위 K개 유지."""
    corr = features.corrwith(target).abs()
    strong = corr[corr > min_corr]
    if len(strong) == 0:
        raise ValueError("타깃과 상관>|{min_corr}| 인 피처가 없습니다.")
    ranked = strong.sort_values(ascending=False).head(top_k)
    print(f"  - 상관>|{min_corr}| 피처 수: {len(strong)} → 상위 {len(ranked)}개 유지")
    return features[ranked.index]


def _remove_feature_multicollinearity(df: pd.DataFrame, threshold: float = 0.9):
    """피처 간 상관이 높은 컬럼 중 하나 제거."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"  - 피처 간 상관>{threshold} 제거: {len(to_drop)}개")
    return df.drop(columns=to_drop)


def _lgbm_importance_rank(features: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> pd.Series:
    """KFold로 LGBM 중요도 평균 산출."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    importances = np.zeros(features.shape[1])

    for train_idx, val_idx in kf.split(features):
        lgbm = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            n_jobs=-1,
            random_state=42,
        )
        lgbm.fit(features.iloc[train_idx], target.iloc[train_idx])
        importances += lgbm.feature_importances_

    importances /= n_splits
    return pd.Series(importances, index=features.columns)


# -------------------------------------------------------------
# Main selection function
# -------------------------------------------------------------


def select_features(
    train_input_path,
    test_input_path,
    labels_path,
    output_dir,
    n_features: int = 300,
    min_corr: float = 0.05,
    top_k: int = 800,
):
    """
    상관관계 분석을 통해 상위 N개의 피처를 선택하고,
    선택된 피처만 포함하는 새로운 train/test 데이터셋을 생성합니다.
    """
    print(f"'{train_input_path}' 파일에서 학습 피처 로딩 중...")
    train_df = pd.read_csv(train_input_path)
    
    print(f"'{labels_path}' 파일에서 타겟(Inhibition) 데이터 로딩 중...")
    labels_df = pd.read_csv(labels_path)

    # 피처 데이터와 타겟 데이터를 'ID' 기준으로 병합
    merged_df = pd.merge(train_df, labels_df[['ID', 'Inhibition']], on='ID')
    
    # --- 피처 선택 ---
    print("피처와 타겟(Inhibition) 간의 상관관계 분석 중...")
    
    # ID, Canonical_Smiles, Inhibition 제외한 모든 열을 피처로 간주
    features = merged_df.drop(columns=['ID', 'Canonical_Smiles', 'Inhibition'])
    # 타겟 변수
    target = merged_df['Inhibition']
    
    # ---------------------------------------------------------
    # 단계별 필터링 & 랭킹
    # ---------------------------------------------------------

    # 0) 기본 전처리: 상수 / NaN
    print("[1] 낮은 분산·고 NaN 피처 제거")
    features = _remove_low_variance(features)
    features = _remove_high_nan(features)

    # 1) 타깃 상관기반 필터
    print("[2] 타깃 상관 필터링")
    features = _correlation_filter(features, target, min_corr=min_corr, top_k=top_k)

    # 2) 다중공선성 제거
    print("[3] 피처 간 상관 제거")
    features = _remove_feature_multicollinearity(features)

    # 3) LightGBM 중요도 랭킹
    print("[4] LightGBM 중요도 기반 상위 피처 선정")
    imp_series = _lgbm_importance_rank(features, target)
    selected_features = imp_series.sort_values(ascending=False).head(n_features).index.tolist()

    print(f"\n최종 선택된 피처 {len(selected_features)}개:")
    print(selected_features[:20], '...' if len(selected_features) > 20 else '')
    
    # --- 새로운 데이터셋 생성 ---
    
    # 필요한 기본 열과 선택된 피처 열만 추출
    columns_to_keep = ['ID', 'Canonical_Smiles', 'Inhibition'] + selected_features
    train_selected_df = merged_df[columns_to_keep]
    
    # Test 데이터셋 로드 및 동일한 피처 선택
    print(f"\n'{test_input_path}' 파일에서 테스트 데이터 로딩 중...")
    test_df = pd.read_csv(test_input_path)
    test_columns_to_keep = ['ID', 'Canonical_Smiles'] + selected_features
    test_selected_df = test_df[test_columns_to_keep]
    
    # --- 결과 저장 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_output_path = os.path.join(output_dir, 'train_selected_features.csv')
    test_output_path = os.path.join(output_dir, 'test_selected_features.csv')
    
    train_selected_df.to_csv(train_output_path, index=False)
    print(f"\n선택된 피처를 포함한 학습 데이터가 '{train_output_path}'에 저장되었습니다.")
    
    test_selected_df.to_csv(test_output_path, index=False)
    print(f"선택된 피처를 포함한 테스트 데이터가 '{test_output_path}'에 저장되었습니다.")

    # 선택된 피처 목록 텍스트도 저장
    feat_list_path = os.path.join(output_dir, 'selected_feature_names.txt')
    with open(feat_list_path, 'w') as f_out:
        for col in selected_features:
            f_out.write(col + '\n')
    print(f"선택된 피처 목록 저장 완료: '{feat_list_path}'")


if __name__ == '__main__':
    # --- 경로 설정 ---
    PROJECT_ROOT = get_project_root()
    
    # generate_features.py의 출력물을 입력으로 사용
    TRAIN_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'CYP3A4_feature_engineering', 'train_features.csv')
    TEST_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'CYP3A4_feature_engineering', 'test_features.csv')
    
    # 타겟(Inhibition) 값이 포함된 원본 학습 데이터 경로
    LABELS_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'train_data.csv')
    
    # 선택된 피처를 저장할 디렉토리
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train_selected')
    
    # --- 피처 선택 실행 ---
    print("="*50)
    print("핵심 피처 선택 시작")
    # 성능 향상을 위해 상위 300개의 피처를 선택
    select_features(
        train_input_path=TRAIN_FEATURES_PATH, 
        test_input_path=TEST_FEATURES_PATH, 
        labels_path=LABELS_PATH,
        output_dir=OUTPUT_DIR, 
        n_features=300
    )
    print("="*50) 