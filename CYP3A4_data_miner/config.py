# CYP3A4_data_miner/config.py

# ChEMBL API를 위한 설정
# 사람의 CYP3A4 효소에 대한 ChEMBL Target ID
TARGET_CHEMBL_ID = "CHEMBL340"

# 데이터 수집 시 필터링할 활성 타입 (IC50 -> Inhibition)
ACTIVITY_TYPE = "Inhibition"

# 저해율(%) 데이터는 단위가 '%'로 고정되므로 nM 단위 필터는 불필요
# STANDARD_UNITS = "nM"


# 데이터 처리 및 저장을 위한 설정
# 최종 데이터셋을 저장할 경로
OUTPUT_DIR = "dataset"
# 최종 생성될 CSV 파일명
OUTPUT_FILENAME = "cyp3a4_regression_dataset.csv"

# 원본 데이터를 저장할 경로 및 파일명
RAW_DATA_DIR = "raw_data"
RAW_DATA_FILENAME = "cyp3a4_raw_activity_data.csv"

# 저해율(%) 데이터를 사용하므로 IC50 상한선 설정은 불필요
# IC50_UPPER_LIMIT = 100000.0 