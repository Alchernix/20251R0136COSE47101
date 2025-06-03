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
    df[col] = df[col].map({'예': 1, '아니오': 0, '모름': np.nan})
    # '모름' 응답은 NaN으로 처리하여 분석에서 제외

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

for idx, row in df_filtered.iterrows():  # 이상치 제거된 데이터 사용
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
            
            # 추천 여부도 함께 저장 (모름 응답은 NaN으로 유지)
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
    # 결측치(모름 응답)는 제외하고 예/아니오 응답만으로 이상치 탐지
    # 결측치가 너무 많은 행은 제외
    valid_rows = df[prereq_columns].dropna(thresh=len(prereq_columns)//2)
    if len(valid_rows) > 10:
        lof_features = valid_rows.copy()
        
        # 각 행별로 결측치를 해당 열의 평균으로 대체
        for col in lof_features.columns:
            lof_features[col].fillna(lof_features[col].mean(), inplace=True)
        
        # LOF 모델 학습
        lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
        outlier_scores = lof.fit_predict(lof_features)
        
        # 원본 데이터프레임에 이상치 점수 매핑
        df['is_outlier'] = False  # 기본값
        df.loc[valid_rows.index, 'is_outlier'] = (outlier_scores == -1)
        
        print("\n이상치로 식별된 응답자 수:", df['is_outlier'].sum())
        
        # 이상치 제거한 데이터 준비
        df_filtered = df[~df['is_outlier']]
        print("이상치 제거 후 데이터 크기:", df_filtered.shape)
    else:
        print("\n유효한 응답이 충분하지 않아 LOF를 적용하지 않음")
        df['is_outlier'] = False
        df_filtered = df.copy()
else:
    print("\n표본 크기가 너무 작아 LOF를 적용하지 않음")
    df['is_outlier'] = False
    df_filtered = df.copy()

# 9. 기술 통계 계산
statistics = []

for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    # 필요한 데이터가 있는 행만 필터링 (모름 응답은 제외)
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
plt.figure(figsize=(14, 8))

# 추천률과 실제 순서 일치율 비교
x = range(len(stats_df))
width = 0.35

plt.bar([i - width/2 for i in x], stats_df['Recommend %'], width, label='추천 비율 (%)', color='skyblue')
plt.bar([i + width/2 for i in x], stats_df['Took A First %'], width, label='실제 순서 일치율 (%)', color='salmon')

plt.xlabel('과목 쌍', fontsize=12)
plt.ylabel('비율 (%)', fontsize=12)
plt.title('선수과목 추천 비율 vs 실제 수강 순서 일치율', fontsize=14)
plt.xticks(x, [f"{row['Course A']}->{row['Course B']}" for _, row in stats_df.iterrows()], rotation=45, ha='right')
plt.legend(fontsize=12)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 표본 크기 정보 추가
for i, v in enumerate(stats_df['Sample Size']):
    plt.text(i, 5, f"n={v}", ha='center', fontsize=9, color='dimgray')

# 통계적 유의성 표시
for i, row in enumerate(stats_df.itertuples()):
    if row.Significant == 'Yes':
        plt.text(i, 100, '*', fontsize=20, ha='center', color='red')

plt.savefig('prereq_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 추가 시각화: 과목 쌍별 추천 비율과 실제 순서 상관관계
plt.figure(figsize=(10, 8))
plt.scatter(stats_df['Recommend %'], stats_df['Took A First %'], 
           s=stats_df['Sample Size']*5, alpha=0.7, c='teal')

# 과목 쌍 레이블 추가
for i, row in stats_df.iterrows():
    plt.annotate(f"{row['Course A']}->{row['Course B']}", 
                xy=(row['Recommend %'], row['Took A First %']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# 45도 선 추가 (완벽한 일치를 나타냄)
max_val = max(stats_df['Recommend %'].max(), stats_df['Took A First %'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

plt.xlabel('추천 비율 (%)', fontsize=12)
plt.ylabel('실제 순서 일치율 (%)', fontsize=12)
plt.title('선수과목 추천 비율과 실제 수강 순서 일치율의 관계', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('prereq_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 상세 교차표 분석 (유의미한 관계가 있는 경우)
for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    # 필요한 데이터가 있는 행만 필터링 (모름 응답은 제외)
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

# 13. 종합 결론
significant_pairs = stats_df[stats_df['Significant'] == 'Yes']
print("\n==== 종합 결론 ====")

if len(significant_pairs) > 0:
    print(f"총 {len(significant_pairs)}개 과목 쌍에서 수강 순서와 추천 사이에 유의미한 관계가 발견되었습니다:")
    for _, row in significant_pairs.iterrows():
        print(f"- {row['Course A']} -> {row['Course B']} (p={row['p-value']}, χ²={row['Chi-Square']})")
    print("\n이러한 과목들은 실제 수강 경험이 선수과목 추천에 영향을 미치는 것으로 보입니다.")
else:
    print("어떤 과목 쌍에서도 수강 순서와 추천 사이에 통계적으로 유의미한 관계가 발견되지 않았습니다.")
    print("이는 선수과목 추천이 실제 수강 경험보다는 다른 요인(예: 교육과정 구조, 과목 난이도 등)에 기반할 수 있음을 시사합니다.")

print("\n추천 비율이 가장 높은 과목 쌍 (응답자의 50% 이상이 선수과목으로 추천):")
top_recommended = stats_df[stats_df['Recommend %'] >= 50].sort_values('Recommend %', ascending=False)
if len(top_recommended) > 0:
    for _, row in top_recommended.iterrows():
        print(f"- {row['Course A']} -> {row['Course B']}: {row['Recommend %']}% (n={row['Sample Size']})")
else:
    print("응답자의 50% 이상이 선수과목으로 추천한 과목 쌍이 없습니다.")

print("\n실제 순서 일치율이 가장 높은 과목 쌍 (80% 이상의 학생이 선수과목 순서로 수강):")
top_ordered = stats_df[stats_df['Took A First %'] >= 80].sort_values('Took A First %', ascending=False)
if len(top_ordered) > 0:
    for _, row in top_ordered.iterrows():
        print(f"- {row['Course A']} -> {row['Course B']}: {row['Took A First %']}% (n={row['Sample Size']})")
else:
    print("80% 이상의 학생이 선수과목 순서로 수강한 과목 쌍이 없습니다.")

print("\n추천 비율과 실제 순서 일치율 간의 차이가 큰 과목 쌍 (25%p 이상):")
discrepancy = stats_df.copy()
discrepancy['Difference'] = abs(discrepancy['Recommend %'] - discrepancy['Took A First %'])
large_diff = discrepancy[discrepancy['Difference'] >= 25].sort_values('Difference', ascending=False)
if len(large_diff) > 0:
    for _, row in large_diff.iterrows():
        print(f"- {row['Course A']} -> {row['Course B']}: 추천 {row['Recommend %']}% vs 실제 순서 {row['Took A First %']}% (차이: {row['Difference']:.1f}%p)")
else:
    print("추천 비율과 실제 순서 일치율 간의 차이가 25%p 이상인 과목 쌍이 없습니다.")

# 전체적인 결론
print("\n결론적으로, ", end="")
if len(significant_pairs) > 0:
    print(f"일부 과목 쌍({len(significant_pairs)}개)에서는 실제 수강 순서와 선수과목 추천 사이에 유의미한 관계가 있지만, ")
    print(f"대부분의 과목 쌍({len(stats_df) - len(significant_pairs)}개)에서는 그런 관계가 발견되지 않았습니다.")
    print("이는 선수과목 추천이 실제 수강 경험 외에도 다양한 요인에 영향을 받을 수 있음을 시사합니다.")
else:
    print("실제 수강 순서와 선수과목 추천 사이에 통계적으로 유의미한 관계가 발견되지 않았습니다.")
    print("이는 학생들의 선수과목 추천이 개인적인 수강 경험보다 교과 내용의 논리적 연결성, 난이도 등 다른 요인에 더 영향을 받을 수 있음을 시사합니다.")