import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 초기 탐색
file_path = '/content/drive/MyDrive/Colab Notebooks/Data Science/데이터과학 설문조사_수강신청추천시스템.csv'
df = pd.read_csv(file_path)

print(f"데이터 크기: {df.shape}")
print("\n처음 5개 행:")
print(df.head(2))  # 처음 2개 행만 출력 (출력 공간 절약)

# 2. 컬럼명 간소화 (작업 편의를 위해)
df.columns = [
    'timestamp', 'major_type', 
    'sem1_1', 'sem1_2', 'sem2_1', 'sem2_2', 'sem3_1', 'sem3_2', 'sem4_1', 'sem4_2',
    'rec_ds_algo', 'rec_ds_algo_others', 'rec_logic_arch', 'rec_arch_os', 'rec_os_sysprog',
    'rec_ai_dl', 'rec_ai_ml', 'rec_theory_pl', 'rec_pl_compiler', 'rec_comm_network',
    'other_prereqs'
]

# 3. 응답 변수 이진화 (예=1, 아니오=0, 모름=NaN)
prereq_columns = [
    'rec_ds_algo', 'rec_ds_algo_others', 'rec_logic_arch', 'rec_arch_os', 'rec_os_sysprog',
    'rec_ai_dl', 'rec_ai_ml', 'rec_theory_pl', 'rec_pl_compiler', 'rec_comm_network'
]

for col in prereq_columns:
    df[col] = df[col].map({'예': 1, '아니오': 0})
    # '모름' 응답은 NaN으로 처리

# 4. 데이터 결측치 확인
missing_values = df[prereq_columns].isnull().sum()
print("\n결측치 개수 (각 추천 변수):")
print(missing_values)

# 학기 컬럼 목록
semester_cols = ['sem1_1', 'sem1_2', 'sem2_1', 'sem2_2', 'sem3_1', 'sem3_2', 'sem4_1', 'sem4_2']

# 5. 과목 리스트 추출 및 표준화 함수
def extract_courses(row, semester_cols):
    courses = {}
    for i, col in enumerate(semester_cols):
        if pd.notna(row[col]) and row[col].strip():
            semester = i + 1  # 1부터 8까지의 학기 값
            course_list = [c.strip() for c in row[col].split(',') if c.strip()]
            for course in course_list:
                courses[course] = semester
    return courses

# 6. 주요 과목 쌍 정의
course_pairs = [
    ('자료구조', '알고리즘', 'rec_ds_algo'),
    ('논리설계', '컴퓨터구조', 'rec_logic_arch'),
    ('컴퓨터구조', '운영체제', 'rec_arch_os'),
    ('운영체제', '시스템프로그래밍', 'rec_os_sysprog'),
    ('인공지능', '딥러닝', 'rec_ai_dl'),
    ('인공지능', '기계학습', 'rec_ai_ml'),
    ('계산이론', '프로그래밍언어', 'rec_theory_pl'),
    ('프로그래밍언어', '컴파일러', 'rec_pl_compiler'),
    ('데이터통신', '컴퓨터네트워크', 'rec_comm_network')
]

# 7. 실제 이수 순서 변수 생성
student_courses = []

for idx, row in df.iterrows():
    courses_by_semester = extract_courses(row, semester_cols)
    student_info = {'student_id': idx}
    
    # 각 과목 쌍에 대해 이수 순서 확인
    for course_a, course_b, rec_col in course_pairs:
        # 두 과목 모두 수강한 경우에만 분석
        if course_a in courses_by_semester and course_b in courses_by_semester:
            sem_a = courses_by_semester[course_a]
            sem_b = courses_by_semester[course_b]
            
            # 순서 변수 생성 (A를 먼저 들었으면 1, 아니면 0)
            took_a_first = 1 if sem_a < sem_b else 0
            
            pair_key = f"{course_a}_{course_b}"
            student_info[f"sem_{course_a}"] = sem_a
            student_info[f"sem_{course_b}"] = sem_b
            student_info[f"took_{pair_key}_first"] = took_a_first
            
            # 추천 여부도 함께 저장
            student_info[f"recommend_{pair_key}"] = row[rec_col]
    
    student_courses.append(student_info)

# 학생별 과목 이수 순서 및 추천 데이터프레임 생성
order_df = pd.DataFrame(student_courses)
print("\n이수 순서 데이터프레임 크기:", order_df.shape)
print("\n이수 순서 데이터프레임 샘플:")
print(order_df.head(2))

# 8. 이상치 탐지 (LOF 적용)
# 추천 응답 패턴만 사용하여 이상치 식별
if len(df) > 10:  # LOF는 최소 10개 이상의 샘플이 필요
    lof_features = df[prereq_columns].fillna(0.5)  # 결측치는 중간값으로 처리
    
    # LOF 모델 학습
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    outlier_scores = lof.fit_predict(lof_features)
    
    # 이상치 점수 저장
    df['outlier_score'] = outlier_scores
    
    # 이상치 식별 (-1은 이상치, 1은 정상)
    df['is_outlier'] = df['outlier_score'] == -1
    
    print("\n이상치로 식별된 응답자 수:", df['is_outlier'].sum())
    
    # 이상치 제거한 데이터 준비
    df_filtered = df[~df['is_outlier']]
    print("이상치 제거 후 데이터 크기:", df_filtered.shape)
else:
    print("\n표본 크기가 너무 작아 LOF를 적용하지 않음")
    df_filtered = df.copy()
    df['is_outlier'] = False

    # 9. 기술 통계 계산
statistics = []

for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    # 필요한 데이터가 있는 행만 필터링
    pair_data = order_df[[took_col, rec_pair_col]].dropna()
    
    if len(pair_data) > 0:
        # 기술 통계
        recommend_pct = pair_data[rec_pair_col].mean() * 100
        took_first_pct = pair_data[took_col].mean() * 100
        
        # 교차표 작성
        if len(pair_data) > 1:  # 최소 2개 이상의 데이터가 필요
            try:
                cross_tab = pd.crosstab(pair_data[took_col], pair_data[rec_pair_col])
                
                # 카이제곱 검정 (기대 빈도가 5 미만인 셀이 있으면 Fisher's exact test 고려)
                chi2, p_value, dof, expected = chi2_contingency(cross_tab)
                
                # 결과 저장
                statistics.append({
                    'Course A': course_a,
                    'Course B': course_b,
                    'Recommend %': round(recommend_pct, 1),
                    'Took A First %': round(took_first_pct, 1),
                    'Sample Size': len(pair_data),
                    'Chi-Square': round(chi2, 2),
                    'p-value': round(p_value, 3),
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except:
                # 교차표에 0행/0열이 있는 경우 오류 처리
                statistics.append({
                    'Course A': course_a,
                    'Course B': course_b,
                    'Recommend %': round(recommend_pct, 1),
                    'Took A First %': round(took_first_pct, 1),
                    'Sample Size': len(pair_data),
                    'Chi-Square': None,
                    'p-value': None,
                    'Significant': 'N/A'
                })
# 10. 결과 출력
stats_df = pd.DataFrame(statistics)
print("\n과목 쌍별 통계 분석 결과:")
print(stats_df)

# 11. 결과 시각화
plt.figure(figsize=(12, 8))

# 추천률과 실제 순서 일치율 비교
x = range(len(stats_df))
width = 0.35

plt.bar([i - width/2 for i in x], stats_df['Recommend %'], width, label='추천 비율 (%)')
plt.bar([i + width/2 for i in x], stats_df['Took A First %'], width, label='실제 순서 일치율 (%)')

plt.xlabel('과목 쌍')
plt.ylabel('비율 (%)')
plt.title('선수과목 추천 비율 vs 실제 수강 순서 일치율')
plt.xticks(x, [f"{row['Course A']}->{row['Course B']}" for _, row in stats_df.iterrows()], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('prereq_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 상세 교차표 분석 (유의미한 관계가 있는 경우)
for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    # 필요한 데이터가 있는 행만 필터링
    pair_data = order_df[[took_col, rec_pair_col]].dropna()
    
    if len(pair_data) > 5:  # 최소한의 데이터가 있는 경우만
        cross_tab = pd.crosstab(
            pair_data[took_col], 
            pair_data[rec_pair_col],
            rownames=['실제 순서 (1=A먼저)'],
            colnames=['추천 (1=추천함)']
        )
        
        # 상대 빈도 계산
        cross_tab_pct = pd.crosstab(
            pair_data[took_col], 
            pair_data[rec_pair_col], 
            normalize='index'
        ) * 100
        
        # 카이제곱 검정
        try:
            chi2, p_value, dof, expected = chi2_contingency(cross_tab)
            
            print(f"\n{course_a} -> {course_b} 관계 분석:")
            print(f"표본 크기: {len(pair_data)}")
            print(f"카이제곱: {chi2:.2f}, p-value: {p_value:.3f}")
            print("\n교차표 (빈도):")
            print(cross_tab)
            print("\n교차표 (행 기준 %):")
            print(cross_tab_pct.round(1))
            
            # p-value가 0.05 미만인 경우 유의미한 관계로 간주
            if p_value < 0.05:
                print(f"결론: {course_a}와 {course_b} 간의 수강 순서와 추천 사이에 유의미한 관계가 있습니다.")
            else:
                print(f"결론: {course_a}와 {course_b} 간의 수강 순서와 추천 사이에 유의미한 관계가 없습니다.")
        except:
            print(f"\n{course_a} -> {course_b}: 충분한 데이터가 없어 통계 분석이 불가능합니다.")