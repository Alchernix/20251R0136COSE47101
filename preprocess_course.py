from paths import course_tag_path
import pandas as pd  # 2) course_tag.xlsx 전처리

#    - header=1  : 엑셀 두 번째 줄을 칼럼명(header)으로 사용 -> 두번째 행 
#    - skiprows  : 세 번째·네 번째 줄(index=2,3)을 건너뜀 (총개수/빈 행 등)

raw_course = pd.read_excel(course_tag_path, header=1, skiprows=[2, 3])

# 전처리 전 컬럼명 확인 (디버깅용)
print("==== 전처리 전 raw_course.columns ====")
print(raw_course.columns.tolist(), "\n")

# "과목코드" 열 자동 감지
#      - 보통 'COSE' 접두사를 포함하므로, Unnamed* 중에서 COSE 패턴이 있는 칼럼을 찾아냄
possible_code_cols = [
    c for c in raw_course.columns
    if c.startswith('Unnamed') and raw_course[c].dropna().astype(str).str.contains('COSE').any()
]
if not possible_code_cols:
    raise RuntimeError("과목코드 열을 자동으로 감지하지 못했습니다. 컬럼명을 직접 확인해주세요.")
course_code_col = possible_code_cols[0]
print(f"감지된 과목코드 컬럼명: '{course_code_col}' → 이후 'Course_Code'로 변경\n")

# "과목명(태그명)" 열 자동 감지
possible_name_cols = [
    c for c in raw_course.columns
    if '태그명' in c or '과목명' in c or 'Course_Name' in c
]
if not possible_name_cols:
    raise RuntimeError("과목명(태그명) 열을 찾지 못했습니다. 컬럼명을 직접 확인해주세요.")
course_name_col = possible_name_cols[0]
print(f"감지된 과목명(한글) 칼럼명: '{course_name_col}' → 이후 'Course_Name'로 변경\n")

# (칼럼명 변경: 과목코드 → 'Course_Code', 과목명 → 'Course_Name'
raw_course = raw_course.rename(columns={course_code_col: 'Course_Code',
                                        course_name_col: 'Course_Name'})

# 실제 태그 컬럼만 추출
#       - 'Course_Code', 'Course_Name'을 제외한 non-Unnamed 칼럼들을 모두 태그로 감지
tag_cols = [
    c for c in raw_course.columns
    if (not c.startswith('Unnamed')) and (c not in ['Course_Code', 'Course_Name'])
]
print("추출된 태그 컬럼(tag_cols) 리스트:")
print(tag_cols, "\n")

# "과목명(Course_Name)"이 NaN인 행 제거
course_df = raw_course[raw_course['Course_Name'].notna()].copy().reset_index(drop=True)

# 필요한 칼럼만 남기기: ['Course_Code', 'Course_Name'] + tag_cols
keep_cols = ['Course_Code', 'Course_Name'] + tag_cols
course_df = course_df[keep_cols].copy()

# 태그 컬럼을 0/1 이진값으로 변환
for t in tag_cols:
    # NaN → 0, 문자열(태그명) → 1
    course_df[t] = course_df[t].notna().astype(int)
