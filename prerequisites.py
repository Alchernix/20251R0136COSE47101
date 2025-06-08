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

# 7. 이상치 탐지 (LOF 적용) - 순서 변경: 이제 먼저 이상치를 탐지합니다
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

# 8. 실제 이수 순서 변수 생성 - 이제 df_filtered를 사용합니다
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



# 1. 학생별 수강 패턴 클러스터링
def analyze_student_patterns(df_filtered, semester_cols, course_pairs):
    """학생별 수강 패턴을 클러스터링하여 유형 분류"""

    def count_theory_courses(courses_dict):
        theory_courses = ['계산이론', '이산수학', '확률및랜덤과정', '수치해석']
        return sum(1 for course in courses_dict.keys() if any(theory in course for theory in theory_courses))

    def count_practical_courses(courses_dict):
        practical_courses = ['프로그래밍', '시스템', '네트워크', '데이터베이스', '소프트웨어공학']
        return sum(1 for course in courses_dict.keys() if any(practical in course for practical in practical_courses))

    def calculate_prerequisite_compliance(courses_dict, course_pairs):
        compliance_count = 0
        total_pairs = 0

        for course_a, course_b, _ in course_pairs:
            if course_a in courses_dict and course_b in courses_dict:
                if courses_dict[course_a] < courses_dict[course_b]:
                    compliance_count += 1
                total_pairs += 1

        return compliance_count / max(total_pairs, 1)

    def calculate_semester_load_variance(row, semester_cols):
        loads = []
        for col in semester_cols:
            if pd.notna(row[col]) and row[col].strip():
                course_count = len([c.strip() for c in row[col].split(',') if c.strip()])
                loads.append(course_count)
            else:
                loads.append(0)
        return np.var(loads) if loads else 0

    student_features = []
    student_ids = []

    for idx, row in df_filtered.iterrows():
        courses_by_semester = extract_courses(row, semester_cols)

        if len(courses_by_semester) == 0:
            continue

        # 학생별 특성 벡터 생성
        total_courses = len(courses_by_semester)
        theory_count = count_theory_courses(courses_by_semester)
        practical_count = count_practical_courses(courses_by_semester)

        features = [
            total_courses,  # 총 수강 과목 수
            theory_count / max(total_courses, 1),  # 이론 과목 비율
            practical_count / max(total_courses, 1),  # 실습 과목 비율
            calculate_prerequisite_compliance(courses_by_semester, course_pairs),  # 선수과목 준수율
            calculate_semester_load_variance(row, semester_cols),  # 학기별 수강 부하 분산
            sum(1 for sem in courses_by_semester.values() if sem <= 2) / max(total_courses, 1),  # 초기 집중도
            sum(1 for sem in courses_by_semester.values() if sem >= 7) / max(total_courses, 1)   # 후기 집중도
        ]

        student_features.append(features)
        student_ids.append(idx)

    if len(student_features) < 3:
        print("클러스터링을 위한 충분한 데이터가 없습니다.")
        return None, None, None

    # K-means 클러스터링
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(student_features)

    # 최적 클러스터 수 결정 (3개로 고정)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # 클러스터별 특성 분석
    cluster_analysis = analyze_cluster_characteristics(df_filtered, clusters, student_ids, course_pairs)

    return clusters, student_ids, cluster_analysis

def analyze_cluster_characteristics(df, clusters, student_ids, course_pairs):
    """클러스터별 특성 및 추천 패턴 분석"""

    prereq_columns = [
        'rec_ds_algo', 'rec_ds_algo_others', 'rec_logic_arch', 'rec_arch_os', 'rec_os_sysprog',
        'rec_ai_dl', 'rec_ai_ml', 'rec_theory_pl', 'rec_pl_compiler', 'rec_comm_network'
    ]

    cluster_patterns = {}

    for cluster_id in range(3):
        cluster_students = [student_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
        cluster_data = df.loc[cluster_students]

        if len(cluster_students) == 0:
            continue

        # 클러스터별 추천 경향 계산
        recommendations = {}
        for col in prereq_columns:
            if col in cluster_data.columns:
                valid_responses = cluster_data[col].dropna()
                if len(valid_responses) > 0:
                    recommendations[col] = valid_responses.mean()
                else:
                    recommendations[col] = np.nan

        # 클러스터 특성 요약
        cluster_patterns[f'Cluster_{cluster_id}'] = {
            'size': len(cluster_students),
            'recommendation_pattern': recommendations,
            'avg_recommendation_rate': np.nanmean(list(recommendations.values())),
            'recommendation_tendency': classify_recommendation_tendency(recommendations)
        }

    return cluster_patterns

def classify_recommendation_tendency(recommendations):
    """추천 성향 분류"""
    valid_recs = [v for v in recommendations.values() if not np.isnan(v)]
    if not valid_recs:
        return "데이터 부족"

    avg_rate = np.mean(valid_recs)
    if avg_rate >= 0.7:
        return "보수적 (선수과목 중시)"
    elif avg_rate <= 0.3:
        return "자유로운 (선수과목 경시)"
    else:
        return "균형적 (상황별 판단)"

# 2. 과목별 필수도 점수 계산
def calculate_necessity_scores(stats_df):
    """다양한 지표를 종합하여 각 선수과목의 필수도 점수 계산"""

    necessity_scores = {}

    for _, row in stats_df.iterrows():
        course_a = row['Course A']
        course_b = row['Course B']
        pair_key = f"{course_a}→{course_b}"

        # 1. 추천 비율 (0-1)
        recommend_rate = row['Recommend %'] / 100

        # 2. 실제 순서 일치율 (0-1)
        actual_order_rate = row['Took A First %'] / 100

        # 3. 표본 크기 가중치 (표본이 클수록 신뢰도 높음)
        sample_size = row['Sample Size']
        sample_weight = min(sample_size / 20, 1.0)  # 20명 이상이면 최대 가중치

        # 4. 일관성 점수 (추천과 실제 순서의 일치 정도)
        consistency = 1 - abs(recommend_rate - actual_order_rate)

        # 5. 종합 필수도 점수 계산
        necessity_score = (
            recommend_rate * 0.4 +           # 추천 비율 40%
            actual_order_rate * 0.3 +       # 실제 순서 30%
            consistency * 0.2 +             # 일관성 20%
            sample_weight * 0.1             # 표본 크기 신뢰도 10%
        )

        # 신뢰도 레벨 계산
        if sample_size >= 25 and consistency >= 0.8:
            confidence = "높음"
        elif sample_size >= 15 and consistency >= 0.6:
            confidence = "보통"
        else:
            confidence = "낮음"

        necessity_scores[pair_key] = {
            'score': necessity_score,
            'components': {
                'recommend_rate': recommend_rate,
                'actual_order_rate': actual_order_rate,
                'consistency': consistency,
                'sample_weight': sample_weight
            },
            'confidence': confidence,
            'sample_size': sample_size
        }

    return necessity_scores

# 3. 시간적 패턴 분석
def analyze_temporal_patterns(df_filtered, semester_cols, course_pairs):
    """선수과목과 후수과목 간의 시간 간격 분석"""

    timing_analysis = {}

    for course_a, course_b, rec_col in course_pairs:
        time_gaps = []
        same_semester_count = 0
        reverse_order_count = 0

        for idx, row in df_filtered.iterrows():
            courses_by_semester = extract_courses(row, semester_cols)

            if course_a in courses_by_semester and course_b in courses_by_semester:
                gap = courses_by_semester[course_b] - courses_by_semester[course_a]
                time_gaps.append(gap)

                if gap == 0:
                    same_semester_count += 1
                elif gap < 0:
                    reverse_order_count += 1

        if time_gaps:
            timing_analysis[f"{course_a}→{course_b}"] = {
                'mean_gap': np.mean(time_gaps),
                'median_gap': np.median(time_gaps),
                'std_gap': np.std(time_gaps),
                'same_semester_rate': same_semester_count / len(time_gaps),
                'reverse_order_rate': reverse_order_count / len(time_gaps),
                'immediate_sequence_rate': sum(1 for gap in time_gaps if gap == 1) / len(time_gaps),
                'total_students': len(time_gaps),
                'gap_distribution': {
                    'negative': sum(1 for gap in time_gaps if gap < 0),
                    'zero': sum(1 for gap in time_gaps if gap == 0),
                    'one': sum(1 for gap in time_gaps if gap == 1),
                    'two_plus': sum(1 for gap in time_gaps if gap >= 2)
                }
            }

    return timing_analysis

# 4. 과목 네트워크 분석
def create_course_dependency_network(order_df, course_pairs):
    """과목 간 의존성 네트워크 생성 및 분석"""

    G = nx.DiGraph()

    # 노드와 엣지 추가
    for course_a, course_b, rec_col in course_pairs:
        took_col = f"took_{course_a}_{course_b}_first"
        rec_pair_col = f"recommend_{course_a}_{course_b}"

        if took_col in order_df.columns and rec_pair_col in order_df.columns:
            pair_data = order_df[[took_col, rec_pair_col]].dropna()

            if len(pair_data) > 0:
                recommendation_strength = pair_data[rec_pair_col].mean()
                actual_strength = pair_data[took_col].mean()
                combined_strength = (recommendation_strength + actual_strength) / 2

                G.add_edge(course_a, course_b,
                          recommendation_strength=recommendation_strength,
                          actual_strength=actual_strength,
                          combined_strength=combined_strength,
                          sample_size=len(pair_data))

    # 네트워크 분석
    if len(G.nodes()) > 0:
        network_analysis = {
            'node_count': len(G.nodes()),
            'edge_count': len(G.edges()),
            'density': nx.density(G),
            'centrality_scores': {
                'in_degree': dict(G.in_degree()),
                'out_degree': dict(G.out_degree()),
            }
        }

        # 연결성이 있는 경우에만 중심성 계산
        if len(G.edges()) > 0:
            try:
                network_analysis['centrality_scores']['betweenness'] = nx.betweenness_centrality(G)
                network_analysis['centrality_scores']['pagerank'] = nx.pagerank(G)
            except:
                network_analysis['centrality_scores']['betweenness'] = {}
                network_analysis['centrality_scores']['pagerank'] = {}

        # 핵심 과목 식별
        out_degrees = dict(G.out_degree())
        network_analysis['core_courses'] = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

        return G, network_analysis
    else:
        return None, None

# 5. 종합 추천 시스템
def create_comprehensive_recommendation_system(necessity_scores, timing_analysis, network_analysis):
    """모든 분석 결과를 종합한 추천 시스템"""

    final_recommendations = {}

    for pair_key, necessity_data in necessity_scores.items():

        # 기본 점수
        base_score = necessity_data['score']

        # 타이밍 적절성 점수
        timing_score = 0.5  # 기본값
        if pair_key in timing_analysis:
            timing_data = timing_analysis[pair_key]
            # 즉시 연속 수강 비율이 높고, 역순 수강 비율이 낮을수록 높은 점수
            timing_score = (
                timing_data['immediate_sequence_rate'] * 0.4 +
                (1 - timing_data['reverse_order_rate']) * 0.4 +
                (1 - timing_data['same_semester_rate']) * 0.2  # 같은 학기 수강은 약간 감점
            )

        # 네트워크 중요도 점수
        network_score = 0.5  # 기본값
        if network_analysis and 'centrality_scores' in network_analysis:
            course_a = pair_key.split('→')[0]
            out_degree_scores = network_analysis['centrality_scores']['out_degree']
            if course_a in out_degree_scores:
                max_out_degree = max(out_degree_scores.values()) if out_degree_scores.values() else 1
                network_score = out_degree_scores[course_a] / max(max_out_degree, 1)

        # 종합 점수 계산 (가중평균)
        final_score = (
            base_score * 0.5 +
            timing_score * 0.3 +
            network_score * 0.2
        )

        # 추천 등급 결정
        if final_score >= 0.75:
            recommendation_level = "강력 추천"
            explanation = "높은 추천률과 실제 순서 준수율을 보임"
        elif final_score >= 0.6:
            recommendation_level = "추천"
            explanation = "상당한 추천률 또는 순서 준수율을 보임"
        elif final_score >= 0.4:
            recommendation_level = "조건부 추천"
            explanation = "일부 지표에서 선수관계의 필요성을 시사"
        else:
            recommendation_level = "선택사항"
            explanation = "선수관계의 필요성이 낮음"

        # 상세 추천 이유 생성
        detailed_reasoning = []

        if necessity_data['components']['recommend_rate'] >= 0.6:
            detailed_reasoning.append(f"학생 추천률 {necessity_data['components']['recommend_rate']:.1%}")

        if necessity_data['components']['actual_order_rate'] >= 0.6:
            detailed_reasoning.append(f"실제 순서 준수율 {necessity_data['components']['actual_order_rate']:.1%}")

        if pair_key in timing_analysis:
            immediate_rate = timing_analysis[pair_key]['immediate_sequence_rate']
            if immediate_rate >= 0.3:
                detailed_reasoning.append(f"연속 학기 수강률 {immediate_rate:.1%}")

        if not detailed_reasoning:
            detailed_reasoning.append("명확한 선수관계 패턴이 관찰되지 않음")

        final_recommendations[pair_key] = {
            'score': final_score,
            'level': recommendation_level,
            'confidence': necessity_data['confidence'],
            'explanation': explanation,
            'detailed_reasoning': "; ".join(detailed_reasoning),
            'sample_size': necessity_data['sample_size'],
            'components': {
                'necessity_score': base_score,
                'timing_score': timing_score,
                'network_score': network_score
            }
        }

    return final_recommendations

# 6. 시각화 함수
def plot_comprehensive_analysis(necessity_scores, timing_analysis, final_recommendations, course_pairs):
    """종합 분석 결과 시각화"""

    # 1. 필수도 점수 vs 타이밍 점수 산점도
    plt.figure(figsize=(15, 12))

    # 서브플롯 1: 필수도 점수 분해
    plt.subplot(2, 3, 1)
    courses = list(necessity_scores.keys())
    recommend_rates = [necessity_scores[course]['components']['recommend_rate'] for course in courses]
    actual_rates = [necessity_scores[course]['components']['actual_order_rate'] for course in courses]

    plt.scatter(recommend_rates, actual_rates, alpha=0.7, s=100)
    for i, course in enumerate(courses):
        plt.annotate(course.replace('→', '\n→'), (recommend_rates[i], actual_rates[i]),
                    fontsize=8, ha='center', va='bottom')

    plt.xlabel('학생 추천 비율')
    plt.ylabel('실제 순서 준수 비율')
    plt.title('추천 vs 실제 순서 패턴')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='완벽한 일치선')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 서브플롯 2: 시간 간격 분포
    plt.subplot(2, 3, 2)
    if timing_analysis:
        immediate_rates = []
        same_sem_rates = []
        course_labels = []

        for course, data in timing_analysis.items():
            immediate_rates.append(data['immediate_sequence_rate'])
            same_sem_rates.append(data['same_semester_rate'])
            course_labels.append(course.replace('→', '\n→'))

        x = np.arange(len(course_labels))
        width = 0.35

        plt.bar(x - width/2, immediate_rates, width, label='연속 학기 수강률', alpha=0.8)
        plt.bar(x + width/2, same_sem_rates, width, label='동일 학기 수강률', alpha=0.8)

        plt.xlabel('과목 쌍')
        plt.ylabel('비율')
        plt.title('수강 시기 패턴 분석')
        plt.xticks(x, course_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 서브플롯 3: 최종 추천 점수
    plt.subplot(2, 3, 3)
    final_scores = [final_recommendations[course]['score'] for course in courses]
    colors = ['red' if score >= 0.75 else 'orange' if score >= 0.6 else 'yellow' if score >= 0.4 else 'lightblue'
              for score in final_scores]

    bars = plt.bar(range(len(courses)), final_scores, color=colors, alpha=0.7)
    plt.xlabel('과목 쌍')
    plt.ylabel('종합 추천 점수')
    plt.title('최종 선수과목 추천 점수')
    plt.xticks(range(len(courses)), [c.replace('→', '\n→') for c in courses], rotation=45, ha='right')

    # 추천 등급별 색상 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='강력 추천 (≥0.75)'),
                      Patch(facecolor='orange', alpha=0.7, label='추천 (≥0.6)'),
                      Patch(facecolor='yellow', alpha=0.7, label='조건부 추천 (≥0.4)'),
                      Patch(facecolor='lightblue', alpha=0.7, label='선택사항 (<0.4)')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)

    # 서브플롯 4: 신뢰도 분석
    plt.subplot(2, 3, 4)
    sample_sizes = [necessity_scores[course]['sample_size'] for course in courses]
    confidence_levels = [necessity_scores[course]['confidence'] for course in courses]

    confidence_colors = {'높음': 'green', '보통': 'orange', '낮음': 'red'}
    colors = [confidence_colors[conf] for conf in confidence_levels]

    plt.scatter(sample_sizes, final_scores, c=colors, alpha=0.7, s=100)
    for i, course in enumerate(courses):
        plt.annotate(course.replace('→', '\n→'), (sample_sizes[i], final_scores[i]),
                    fontsize=8, ha='center', va='bottom')

    plt.xlabel('표본 크기')
    plt.ylabel('최종 추천 점수')
    plt.title('표본 크기 vs 추천 점수 (신뢰도별)')

    # 신뢰도 범례
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='높음'),
                      Patch(facecolor='orange', alpha=0.7, label='보통'),
                      Patch(facecolor='red', alpha=0.7, label='낮음')]
    plt.legend(handles=legend_elements, title='신뢰도', loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    # 서브플롯 5: 추천 등급 분포
    plt.subplot(2, 3, 5)
    recommendation_levels = [final_recommendations[course]['level'] for course in courses]
    level_counts = pd.Series(recommendation_levels).value_counts()

    plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('추천 등급 분포')

    # 서브플롯 6: 점수 구성 요소 분석
    plt.subplot(2, 3, 6)
    necessity_components = [final_recommendations[course]['components']['necessity_score'] for course in courses]
    timing_components = [final_recommendations[course]['components']['timing_score'] for course in courses]
    network_components = [final_recommendations[course]['components']['network_score'] for course in courses]

    x = np.arange(len(courses))
    width = 0.25

    plt.bar(x - width, necessity_components, width, label='필수도 점수', alpha=0.8)
    plt.bar(x, timing_components, width, label='타이밍 점수', alpha=0.8)
    plt.bar(x + width, network_components, width, label='네트워크 점수', alpha=0.8)

    plt.xlabel('과목 쌍')
    plt.ylabel('점수')
    plt.title('점수 구성 요소 분석')
    plt.xticks(x, [c.replace('→', '\n→') for c in courses], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 7. 상세 결과 출력 함수
def print_detailed_results(final_recommendations, cluster_analysis, timing_analysis):
    """상세 분석 결과 출력"""

    print("="*80)
    print("📊 종합 선수과목 추천 분석 결과")
    print("="*80)

    # 1. 최종 추천 결과
    print("\n🎯 **최종 선수과목 추천 순위**")
    sorted_recommendations = sorted(final_recommendations.items(),
                                  key=lambda x: x[1]['score'], reverse=True)

    for i, (course_pair, data) in enumerate(sorted_recommendations, 1):
        print(f"\n{i}. **{course_pair}**")
        print(f"   └ 추천 등급: {data['level']} (점수: {data['score']:.3f})")
        print(f"   └ 신뢰도: {data['confidence']} (표본 크기: {data['sample_size']}명)")
        print(f"   └ 근거: {data['detailed_reasoning']}")

    # 2. 클러스터 분석 결과
    if cluster_analysis:
        print(f"\n👥 **학생 유형별 분석** (총 {sum(cluster['size'] for cluster in cluster_analysis.values())}명 분석)")
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"\n📍 {cluster_name} ({cluster_data['size']}명) - {cluster_data['recommendation_tendency']}")
            print(f"   └ 평균 추천률: {cluster_data['avg_recommendation_rate']:.1%}")

    # 3. 시간적 패턴 분석
    if timing_analysis:
        print(f"\n⏰ **수강 시기 패턴 분석**")

        # 가장 연속적으로 수강하는 과목 쌍
        most_sequential = max(timing_analysis.items(),
                            key=lambda x: x[1]['immediate_sequence_rate'])
        print(f"   🔄 가장 연속적: {most_sequential[0]} "
              f"({most_sequential[1]['immediate_sequence_rate']:.1%}가 연속 학기 수강)")

        # 가장 같은 학기에 수강하는 과목 쌍
        most_concurrent = max(timing_analysis.items(),
                            key=lambda x: x[1]['same_semester_rate'])
        print(f"   ⚡ 가장 동시적: {most_concurrent[0]} "
              f"({most_concurrent[1]['same_semester_rate']:.1%}가 동일 학기 수강)")

        # 역순 수강이 많은 과목 쌍
        most_reverse = max(timing_analysis.items(),
                         key=lambda x: x[1]['reverse_order_rate'])
        if most_reverse[1]['reverse_order_rate'] > 0.1:
            print(f"   🔄 역순 수강 주의: {most_reverse[0]} "
                  f"({most_reverse[1]['reverse_order_rate']:.1%}가 역순 수강)")

    # 4. 핵심 인사이트
    print(f"\n💡 **핵심 인사이트**")

    strong_recommendations = [course for course, data in final_recommendations.items()
                            if data['level'] == '강력 추천']
    if strong_recommendations:
        print(f"   ✅ 강력 추천 과목 쌍: {', '.join(strong_recommendations)}")

    optional_recommendations = [course for course, data in final_recommendations.items()
                              if data['level'] == '선택사항']
    if optional_recommendations:
        print(f"   🤔 유연한 수강 가능: {', '.join(optional_recommendations)}")

    low_confidence = [course for course, data in final_recommendations.items()
                     if data['confidence'] == '낮음']
    if low_confidence:
        print(f"   ⚠️  추가 데이터 필요: {', '.join(low_confidence)}")

# 메인 실행 함수 (기존 코드 뒤에 추가)
def run_advanced_analysis():
    """고급 분석 전체 실행"""

    print("\n" + "="*60)
    print("🚀 고급 선수과목 분석 시작")
    print("="*60)

    # 1. 학생별 수강 패턴 클러스터링
    print("\n1️⃣ 학생 수강 패턴 클러스터링...")
    clusters, student_ids, cluster_analysis = analyze_student_patterns(df_filtered, semester_cols, course_pairs)

    if cluster_analysis:
        print(f"   ✅ {len([c for c in cluster_analysis.values()])}개 클러스터 생성 완료")
    else:
        print("   ⚠️ 클러스터링 데이터 부족")

    # 2. 필수도 점수 계산
    print("\n2️⃣ 과목별 필수도 점수 계산...")
    necessity_scores = calculate_necessity_scores(stats_df)
    print(f"   ✅ {len(necessity_scores)}개 과목 쌍 분석 완료")

    # 3. 시간적 패턴 분석
    print("\n3️⃣ 수강 시기 패턴 분석...")
    timing_analysis = analyze_temporal_patterns(df_filtered, semester_cols, course_pairs)
    print(f"   ✅ {len(timing_analysis)}개 과목 쌍 시간 패턴 분석 완료")

    # 4. 네트워크 분석
    print("\n4️⃣ 과목 의존성 네트워크 분석...")
    network_graph, network_analysis = create_course_dependency_network(order_df, course_pairs)
    if network_analysis:
        print(f"   ✅ {network_analysis['node_count']}개 노드, {network_analysis['edge_count']}개 엣지 네트워크 생성")
    else:
        print("   ⚠️ 네트워크 생성 실패")

    # 5. 종합 추천 시스템
    print("\n5️⃣ 종합 추천 시스템 구축...")
    final_recommendations = create_comprehensive_recommendation_system(
        necessity_scores, timing_analysis, network_analysis)
    print(f"   ✅ {len(final_recommendations)}개 과목 쌍 최종 추천 완료")

    # 6. 결과 출력
    print_detailed_results(final_recommendations, cluster_analysis, timing_analysis)

    # 7. 시각화
    print(f"\n📊 결과 시각화 생성 중...")
    plot_comprehensive_analysis(necessity_scores, timing_analysis, final_recommendations, course_pairs)

    return {
        'necessity_scores': necessity_scores,
        'timing_analysis': timing_analysis,
        'network_analysis': network_analysis,
        'final_recommendations': final_recommendations,
        'cluster_analysis': cluster_analysis
    }

# 추가 분석 함수들

def analyze_course_difficulty_proxy():
    """과목별 난이도 대리 지표 분석"""

    print("\n📈 **과목별 특성 분석**")

    # 과목별 수강 학기 분포 분석
    course_semester_stats = {}

    for _, row in df_filtered.iterrows():
        courses = extract_courses(row, semester_cols)
        for course, semester in courses.items():
            if course not in course_semester_stats:
                course_semester_stats[course] = []
            course_semester_stats[course].append(semester)

    # 과목별 통계 계산
    course_analysis = {}
    for course, semesters in course_semester_stats.items():
        if len(semesters) >= 3:  # 최소 3명 이상 수강한 과목만
            course_analysis[course] = {
                'popularity': len(semesters),
                'avg_semester': np.mean(semesters),
                'semester_std': np.std(semesters),
                'early_adoption_rate': sum(1 for s in semesters if s <= 2) / len(semesters),
                'late_adoption_rate': sum(1 for s in semesters if s >= 7) / len(semesters)
            }

    # 결과 출력
    print("\n🎯 **인기 과목 TOP 5**")
    popular_courses = sorted(course_analysis.items(),
                           key=lambda x: x[1]['popularity'], reverse=True)[:5]
    for i, (course, stats) in enumerate(popular_courses, 1):
        print(f"{i}. {course}: {stats['popularity']}명 수강 "
              f"(평균 {stats['avg_semester']:.1f}학기)")

    print("\n📚 **기초 과목 (이른 시기 수강)**")
    early_courses = sorted(course_analysis.items(),
                          key=lambda x: x[1]['avg_semester'])[:5]
    for course, stats in early_courses:
        print(f"   • {course}: 평균 {stats['avg_semester']:.1f}학기 "
              f"(초기 수강률: {stats['early_adoption_rate']:.1%})")

    print("\n🎓 **고급 과목 (늦은 시기 수강)**")
    late_courses = sorted(course_analysis.items(),
                         key=lambda x: x[1]['avg_semester'], reverse=True)[:5]
    for course, stats in late_courses:
        print(f"   • {course}: 평균 {stats['avg_semester']:.1f}학기 "
              f"(후기 수강률: {stats['late_adoption_rate']:.1%})")

    return course_analysis

def create_prerequisite_recommendation_table(final_recommendations):
    """최종 추천 결과를 표 형태로 정리"""

    # 데이터프레임 생성
    recommendation_data = []

    for course_pair, data in final_recommendations.items():
        course_a, course_b = course_pair.split('→')

        recommendation_data.append({
            '선수과목': course_a,
            '후수과목': course_b,
            '추천등급': data['level'],
            '종합점수': round(data['score'], 3),
            '신뢰도': data['confidence'],
            '표본크기': data['sample_size'],
            '필수도점수': round(data['components']['necessity_score'], 3),
            '타이밍점수': round(data['components']['timing_score'], 3),
            '네트워크점수': round(data['components']['network_score'], 3),
            '상세근거': data['detailed_reasoning']
        })

    recommendation_df = pd.DataFrame(recommendation_data)
    recommendation_df = recommendation_df.sort_values('종합점수', ascending=False)

    print("\n📋 **최종 선수과목 추천 테이블**")
    print("="*120)

    # 테이블 출력 (보기 좋게 정렬)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(recommendation_df.to_string(index=False))

    # 등급별 요약
    print(f"\n📊 **추천 등급별 요약**")
    level_summary = recommendation_df['추천등급'].value_counts()
    for level, count in level_summary.items():
        percentage = count / len(recommendation_df) * 100
        print(f"   • {level}: {count}개 ({percentage:.1f}%)")

    return recommendation_df

def generate_practical_guidelines(final_recommendations, timing_analysis):
    """실용적인 수강 가이드라인 생성"""

    print("\n" + "="*80)
    print("📚 **실용적 수강 가이드라인**")
    print("="*80)

    # 1. 반드시 순서를 지켜야 할 과목들
    critical_sequences = [
        (course, data) for course, data in final_recommendations.items()
        if data['level'] == '강력 추천' and data['confidence'] in ['높음', '보통']
    ]

    if critical_sequences:
        print("\n🚨 **반드시 순서를 지켜야 할 과목 쌍**")
        for course_pair, data in critical_sequences:
            course_a, course_b = course_pair.split('→')
            print(f"   ✅ {course_a} → {course_b}")
            print(f"      └ 근거: {data['detailed_reasoning']}")

    # 2. 유연하게 수강 가능한 과목들
    flexible_courses = [
        (course, data) for course, data in final_recommendations.items()
        if data['level'] in ['선택사항', '조건부 추천']
    ]

    if flexible_courses:
        print(f"\n🔄 **유연한 수강이 가능한 과목 쌍**")
        for course_pair, data in flexible_courses:
            course_a, course_b = course_pair.split('→')
            print(f"   💡 {course_a} ↔ {course_b} (순서 무관)")

    # 3. 수강신청 전략
    print(f"\n🎯 **수강신청 전략 제안**")

    if timing_analysis:
        # 동일 학기 수강이 많은 과목들
        concurrent_courses = [
            (course, data) for course, data in timing_analysis.items()
            if data['same_semester_rate'] >= 0.3
        ]

        if concurrent_courses:
            print(f"\n   📅 **동시 수강 고려 과목들**")
            for course_pair, data in concurrent_courses:
                print(f"      • {course_pair}: {data['same_semester_rate']:.1%}가 동일 학기 수강")
                print(f"        → 수강신청 실패 시 동시 수강 고려")

        # 연속 학기 수강 추천
        sequential_courses = [
            (course, data) for course, data in timing_analysis.items()
            if data['immediate_sequence_rate'] >= 0.4
        ]

        if sequential_courses:
            print(f"\n   ⏭️ **연속 학기 수강 추천**")
            for course_pair, data in sequential_courses:
                print(f"      • {course_pair}: {data['immediate_sequence_rate']:.1%}가 연속 학기 수강")
                print(f"        → 가능하면 연속 학기에 수강하는 것이 유리")

    # 4. 주의사항
    low_confidence_courses = [
        (course, data) for course, data in final_recommendations.items()
        if data['confidence'] == '낮음'
    ]

    if low_confidence_courses:
        print(f"\n⚠️ **주의사항 (데이터 부족으로 신뢰도 낮음)**")
        for course_pair, data in low_confidence_courses:
            print(f"   • {course_pair} (표본: {data['sample_size']}명)")
            print(f"     → 개별 상담 또는 추가 정보 수집 권장")

# 기존 코드에 추가할 실행 부분
print("\n" + "#"*80)
print("# 고급 분석 실행 중...")
print("#"*80)

# 고급 분석 실행
advanced_results = run_advanced_analysis()

# 추가 분석들
course_difficulty_analysis = analyze_course_difficulty_proxy()
recommendation_table = create_prerequisite_recommendation_table(advanced_results['final_recommendations'])
generate_practical_guidelines(advanced_results['final_recommendations'], advanced_results['timing_analysis'])

print("\n" + "🎉 "*20)
print("모든 분석이 완료되었습니다!")
print("🎉 "*20)