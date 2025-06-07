from paths import user_data_path
import pandas as pd

# 3) user_data.xlsx 전처리
#    - “학기에 들은 컴퓨터학과 전공 과목” 칼럼들을 모두 찾아 수강 과목 리스트로 합침

raw_user = pd.read_excel(user_data_path)

# 전처리 전 컬럼명 확인
print("==== 전처리 전 raw_user.columns ====")
print(raw_user.columns.tolist(), "\n")

# '학기에 들은' 이라는 문구가 포함된 칼럼명 모두 수집
semester_cols = [c for c in raw_user.columns if '학기에 들은' in c]
if not semester_cols:
    raise RuntimeError("user_data.xlsx에서 '학기에 들은' 과목 칼럼을 찾지 못했습니다.")
print("감지된 '학기에 들은' 칼럼들:")
for c in semester_cols:
    print("   -", c)
print()

# 사용자가 수강한 과목을 모아 리스트로 반환하는 함수 정의
def collect_taken_courses(row):
    taken = []
    for col in semester_cols:
        val = row[col]
        if isinstance(val, str) and val.strip():
            # 쉼표(,) 기준으로 분리하되, 빈 문자열 제외
            courses = [x.strip() for x in val.split(',') if x.strip()]
            taken.extend(courses)
    # 중복 제거
    return list(set(taken))

# 새로운 DataFrame 생성 및 'taken_courses' 컬럼 추가
user_df_processed = raw_user.copy()
user_df_processed['taken_courses'] = user_df_processed.apply(collect_taken_courses, axis=1)

# 학기별 원본 칼럼들을 유지하거나, 필요없으면 삭제 가능
#       아래는 예시로 학기별 칼럼을 모두 삭제하는 경우:
# user_df_processed = user_df_processed.drop(columns=semester_cols)
