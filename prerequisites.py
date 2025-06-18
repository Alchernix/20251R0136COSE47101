import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
file_path = '/content/drive/MyDrive/Colab Notebooks/Data Science/preprocessed.csv'
df = pd.read_csv(file_path)

print(f"데이터 크기: {df.shape}")

df.columns = [
    'timestamp', 'major_type', 
    'sem1_1', 'sem1_2', 'sem2_1', 'sem2_2', 'sem3_1', 'sem3_2', 'sem4_1', 'sem4_2',
    'rec_ds_algo', 'rec_ds_algo_others', 'rec_logic_arch', 'rec_arch_os', 'rec_os_sysprog',
    'rec_ai_dl', 'rec_ai_ml', 'rec_theory_pl', 'rec_pl_compiler', 'rec_comm_network',
    'other_prereqs'
]

prereq_columns = [
    'rec_ds_algo', 'rec_ds_algo_others', 'rec_logic_arch', 'rec_arch_os', 'rec_os_sysprog',
    'rec_ai_dl', 'rec_ai_ml', 'rec_theory_pl', 'rec_pl_compiler', 'rec_comm_network'
]

for col in prereq_columns:
    df[col] = df[col].map({'예': 1, '아니오': 0, '모름': np.nan})

missing_values = df[prereq_columns].isnull().sum()
print("\n결측치 개수 (각 추천 변수):")
print(missing_values)

semester_cols = ['sem1_1', 'sem1_2', 'sem2_1', 'sem2_2', 'sem3_1', 'sem3_2', 'sem4_1', 'sem4_2']

# 과목 리스트 추출 및 표준화 함수
def extract_courses(row, semester_cols):
    courses = {}
    for i, col in enumerate(semester_cols):
        if pd.notna(row[col]) and row[col].strip():
            semester = i + 1
            course_list = [c.strip() for c in row[col].split(',') if c.strip()]
            for course in course_list:
                courses[course] = semester
    return courses

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

# 이상치 탐지 (LOF 적용)
if len(df) > 10:
    valid_rows = df[prereq_columns].dropna(thresh=len(prereq_columns)//2)
    if len(valid_rows) > 10:
        lof_features = valid_rows.copy()
        
        for col in lof_features.columns:
            lof_features[col].fillna(lof_features[col].mean(), inplace=True)
        lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
        outlier_scores = lof.fit_predict(lof_features)
        
        df['is_outlier'] = False
        df.loc[valid_rows.index, 'is_outlier'] = (outlier_scores == -1)
        
        print("\n이상치로 식별된 응답자 수:", df['is_outlier'].sum())
        
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

# 실제 이수 순서 변수 생성
student_courses = []

for idx, row in df_filtered.iterrows():
    courses_by_semester = extract_courses(row, semester_cols)
    student_info = {'student_id': idx}
    
    # 각 과목 쌍에 대해 이수 순서 확인
    for course_a, course_b, rec_col in course_pairs:
        if course_a in courses_by_semester and course_b in courses_by_semester:
            sem_a = courses_by_semester[course_a]
            sem_b = courses_by_semester[course_b]
            
            # 순서 변수 생성
            took_a_first = 1 if sem_a < sem_b else 0
            
            pair_key = f"{course_a}_{course_b}"
            student_info[f"sem_{course_a}"] = sem_a
            student_info[f"sem_{course_b}"] = sem_b
            student_info[f"took_{pair_key}_first"] = took_a_first
            student_info[f"recommend_{pair_key}"] = row[rec_col]
    
    student_courses.append(student_info)


order_df = pd.DataFrame(student_courses)
print("\n이수 순서 데이터프레임 크기:", order_df.shape)


# 기술 통계 계산
statistics = []

for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    pair_data = order_df[[took_col, rec_pair_col]].dropna()
    
    if len(pair_data) > 0:
        recommend_pct = pair_data[rec_pair_col].mean() * 100
        took_first_pct = pair_data[took_col].mean() * 100
        
        # 교차표 작성
        if len(pair_data) > 1:
            try:
                cross_tab = pd.crosstab(pair_data[took_col], pair_data[rec_pair_col])
                
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

# 결과 출력
stats_df = pd.DataFrame(statistics)
print(stats_df)

plt.figure(figsize=(14, 8))

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

for i, v in enumerate(stats_df['Sample Size']):
    plt.text(i, 5, f"n={v}", ha='center', fontsize=9, color='dimgray')

for i, row in enumerate(stats_df.itertuples()):
    if row.Significant == 'Yes':
        plt.text(i, 100, '*', fontsize=20, ha='center', color='red')

plt.savefig('prereq_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(stats_df['Recommend %'], stats_df['Took A First %'], 
           s=stats_df['Sample Size']*5, alpha=0.7, c='teal')

for i, row in stats_df.iterrows():
    plt.annotate(f"{row['Course A']}->{row['Course B']}", 
                xy=(row['Recommend %'], row['Took A First %']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)


max_val = max(stats_df['Recommend %'].max(), stats_df['Took A First %'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

plt.xlabel('추천 비율 (%)', fontsize=12)
plt.ylabel('실제 순서 일치율 (%)', fontsize=12)
plt.title('선수과목 추천 비율과 실제 수강 순서 일치율의 관계', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('prereq_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 상세 교차표 분석
for course_a, course_b, rec_col in course_pairs:
    pair_key = f"{course_a}_{course_b}"
    took_col = f"took_{pair_key}_first"
    rec_pair_col = f"recommend_{pair_key}"
    
    pair_data = order_df[[took_col, rec_pair_col]].dropna()
    
    if len(pair_data) > 5:
        cross_tab = pd.crosstab(
            pair_data[took_col], 
            pair_data[rec_pair_col],
            rownames=['실제 순서 (1=A먼저)'],
            colnames=['추천 (1=추천함)']
        )
        
        cross_tab_pct = pd.crosstab(
            pair_data[took_col], 
            pair_data[rec_pair_col], 
            normalize='index'
        ) * 100
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(cross_tab)
            
            print(f"\n{course_a} -> {course_b} 관계 분석:")
            print(f"표본 크기: {len(pair_data)}")
            print(f"카이제곱: {chi2:.2f}, p-value: {p_value:.3f}")
            print("\n교차표 (빈도):")
            print(cross_tab)
            print("\n교차표 (행 기준 %):")
            print(cross_tab_pct.round(1))
            
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
    print("어떤 과목 쌍에서도 수강 순서와 추천 사이에 유의미한 관계가 발견되지 않았습니다.")
    print("선수과목 추천이 실제 수강 경험보다는 다른 요인에 기반할 수 있음을 시사.")

print("\n응답자의 50% 이상이 선수과목으로 추천:")
top_recommended = stats_df[stats_df['Recommend %'] >= 50].sort_values('Recommend %', ascending=False)
if len(top_recommended) > 0:
    for _, row in top_recommended.iterrows():
        print(f"- {row['Course A']} -> {row['Course B']}: {row['Recommend %']}% (n={row['Sample Size']})")
else:
    print("응답자의 50% 이상이 선수과목으로 추천한 과목 쌍이 없습니다.")

print("\n80% 이상의 학생이 선수과목 순서로 수강:")
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


print("\n결론, ", end="")
if len(significant_pairs) > 0:
    print(f"일부 과목 쌍({len(significant_pairs)}개)에서는 실제 수강 순서와 선수과목 추천 사이에 유의미한 관계가 있지만, ")
    print(f"대부분의 과목 쌍({len(stats_df) - len(significant_pairs)}개)에서는 그런 관계가 발견되지 않았습니다.")
else:
    print("실제 수강 순서와 선수과목 추천 사이에 유의미한 관계가 발견되지 않았습니다.")



# 학생별 수강 패턴 클러스터링
def analyze_student_patterns(df_filtered, semester_cols, course_pairs):

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

        total_courses = len(courses_by_semester)
        theory_count = count_theory_courses(courses_by_semester)
        practical_count = count_practical_courses(courses_by_semester)

        features = [
            total_courses,
            theory_count / max(total_courses, 1),
            practical_count / max(total_courses, 1),
            calculate_prerequisite_compliance(courses_by_semester, course_pairs),
            calculate_semester_load_variance(row, semester_cols),
            sum(1 for sem in courses_by_semester.values() if sem <= 2) / max(total_courses, 1),  # 초기 집중도
            sum(1 for sem in courses_by_semester.values() if sem >= 7) / max(total_courses, 1)   # 후기 집중도
        ]

        student_features.append(features)
        student_ids.append(idx)

    if len(student_features) < 3:
        print("클러스터링을 위한 충분한 데이터가 없습니다.")
        return None, None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(student_features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    cluster_analysis = analyze_cluster_characteristics(df_filtered, clusters, student_ids, course_pairs)

    return clusters, student_ids, cluster_analysis

def analyze_cluster_characteristics(df, clusters, student_ids, course_pairs):
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

        recommendations = {}
        for col in prereq_columns:
            if col in cluster_data.columns:
                valid_responses = cluster_data[col].dropna()
                if len(valid_responses) > 0:
                    recommendations[col] = valid_responses.mean()
                else:
                    recommendations[col] = np.nan

        cluster_patterns[f'Cluster_{cluster_id}'] = {
            'size': len(cluster_students),
            'recommendation_pattern': recommendations,
            'avg_recommendation_rate': np.nanmean(list(recommendations.values())),
            'recommendation_tendency': classify_recommendation_tendency(recommendations)
        }

    return cluster_patterns

def classify_recommendation_tendency(recommendations):
    valid_recs = [v for v in recommendations.values() if not np.isnan(v)]
    if not valid_recs:
        return "데이터 부족"

    avg_rate = np.mean(valid_recs)
    if avg_rate >= 0.7:
        return "선수과목 매우 중요"
    elif avg_rate <= 0.3:
        return "선수과목 적당히 중요"
    else:
        return "상황별 판단 필요"


# 과목별 필수도 점수 계산
def calculate_necessity_scores(stats_df):
    necessity_scores = {}

    for _, row in stats_df.iterrows():
        course_a = row['Course A']
        course_b = row['Course B']
        pair_key = f"{course_a}→{course_b}"

        # 추천 비율 (0-1)과 실제 순서 일치율 (0-1)
        recommend_rate = row['Recommend %'] / 100
        actual_order_rate = row['Took A First %'] / 100

        sample_size = row['Sample Size']
        sample_weight = min(sample_size / 20, 1.0)

        consistency = 1 - abs(recommend_rate - actual_order_rate)

        necessity_score = (
            recommend_rate * 0.4 +
            actual_order_rate * 0.3 +
            consistency * 0.2 +
            sample_weight * 0.1
        )

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

# 시간적 패턴
def analyze_temporal_patterns(df_filtered, semester_cols, course_pairs):
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



# 과목 네트워크 분석
def create_course_dependency_network(order_df, course_pairs):
    G = nx.DiGraph()

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

        if len(G.edges()) > 0:
            try:
                network_analysis['centrality_scores']['betweenness'] = nx.betweenness_centrality(G)
                network_analysis['centrality_scores']['pagerank'] = nx.pagerank(G)
            except:
                network_analysis['centrality_scores']['betweenness'] = {}
                network_analysis['centrality_scores']['pagerank'] = {}

        out_degrees = dict(G.out_degree())
        network_analysis['core_courses'] = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

        return G, network_analysis
    else:
        return None, None

# 종합 추천 시스템
def create_comprehensive_recommendation_system(necessity_scores, timing_analysis, network_analysis):
    final_recommendations = {}

    for pair_key, necessity_data in necessity_scores.items():

        base_score = necessity_data['score']

        timing_score = 0.5
        if pair_key in timing_analysis:
            timing_data = timing_analysis[pair_key]
            timing_score = (
                timing_data['immediate_sequence_rate'] * 0.4 +
                (1 - timing_data['reverse_order_rate']) * 0.4 +
                (1 - timing_data['same_semester_rate']) * 0.2
            )

        # 네트워크 중요도 점수
        network_score = 0.5
        if network_analysis and 'centrality_scores' in network_analysis:
            course_a = pair_key.split('→')[0]
            out_degree_scores = network_analysis['centrality_scores']['out_degree']
            if course_a in out_degree_scores:
                max_out_degree = max(out_degree_scores.values()) if out_degree_scores.values() else 1
                network_score = out_degree_scores[course_a] / max(max_out_degree, 1)

        # 종합 점수 계산
        final_score = (
            base_score * 0.5 +
            timing_score * 0.3 +
            network_score * 0.2
        )

        if final_score >= 0.75:
            recommendation_level = "강력 추천"
            explanation = "높은 추천률/순서 준수율"
        elif final_score >= 0.6:
            recommendation_level = "추천"
            explanation = "좋은 추천률/순서 준수율"
        elif final_score >= 0.4:
            recommendation_level = "조건부 추천"
            explanation = "일부 지표에서 선수관계의 필요"
        else:
            recommendation_level = "선택사항"
            explanation = "선수과목 효과 적음"

        detailed_reasoning = []

        if necessity_data['components']['recommend_rate'] >= 0.6:
            detailed_reasoning.append(f"학생 추천률 {necessity_data['components']['recommend_rate']:.1%}")

        if necessity_data['components']['actual_order_rate'] >= 0.6:
            detailed_reasoning.append(f"순서 준수율 {necessity_data['components']['actual_order_rate']:.1%}")

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


def plot_comprehensive_analysis(necessity_scores, timing_analysis, final_recommendations, course_pairs):
    plt.figure(figsize=(15, 12))

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

    # 시간 간격 분포
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

    # 최종 추천 점수
    plt.subplot(2, 3, 3)
    final_scores = [final_recommendations[course]['score'] for course in courses]
    colors = ['red' if score >= 0.75 else 'orange' if score >= 0.6 else 'yellow' if score >= 0.4 else 'lightblue'
              for score in final_scores]

    bars = plt.bar(range(len(courses)), final_scores, color=colors, alpha=0.7)
    plt.xlabel('과목 쌍')
    plt.ylabel('종합 추천 점수')
    plt.title('최종 선수과목 추천 점수')
    plt.xticks(range(len(courses)), [c.replace('→', '\n→') for c in courses], rotation=45, ha='right')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='강력 추천 (≥0.75)'),
                      Patch(facecolor='orange', alpha=0.7, label='추천 (≥0.6)'),
                      Patch(facecolor='yellow', alpha=0.7, label='조건부 추천 (≥0.4)'),
                      Patch(facecolor='lightblue', alpha=0.7, label='선택사항 (<0.4)')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)

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

    legend_elements = [Patch(facecolor='green', alpha=0.7, label='높음'),
                      Patch(facecolor='orange', alpha=0.7, label='보통'),
                      Patch(facecolor='red', alpha=0.7, label='낮음')]
    plt.legend(handles=legend_elements, title='신뢰도', loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    recommendation_levels = [final_recommendations[course]['level'] for course in courses]
    level_counts = pd.Series(recommendation_levels).value_counts()

    plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('추천 등급 분포')

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

# 상세 결과 출력 함수
def print_detailed_results(final_recommendations, cluster_analysis, timing_analysis):

    sorted_recommendations = sorted(final_recommendations.items(),
                                  key=lambda x: x[1]['score'], reverse=True)

    for i, (course_pair, data) in enumerate(sorted_recommendations, 1):
        print(f"\n{i}. **{course_pair}**")
        print(f"   └ 추천 등급: {data['level']} (점수: {data['score']:.3f})")
        print(f"   └ 신뢰도: {data['confidence']} (표본 크기: {data['sample_size']}명)")
        print(f"   └ 근거: {data['detailed_reasoning']}")

    if cluster_analysis:
        print(f"\n학생 유형별 분석 (총 {sum(cluster['size'] for cluster in cluster_analysis.values())}명 분석)")
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"\n {cluster_name} ({cluster_data['size']}명) - {cluster_data['recommendation_tendency']}")
            print(f"   └ 평균 추천률: {cluster_data['avg_recommendation_rate']:.1%}")

    # 시간적 패턴 분석
    if timing_analysis:
        print(f"\n 수강 시기 패턴 분석")

        most_sequential = max(timing_analysis.items(),
                            key=lambda x: x[1]['immediate_sequence_rate'])
        print(f"    가장 연속적: {most_sequential[0]} "
              f"({most_sequential[1]['immediate_sequence_rate']:.1%}가 연속 학기 수강)")

        most_concurrent = max(timing_analysis.items(),
                            key=lambda x: x[1]['same_semester_rate'])
        print(f"    가장 동시적: {most_concurrent[0]} "
              f"({most_concurrent[1]['same_semester_rate']:.1%}가 동일 학기 수강)")

        most_reverse = max(timing_analysis.items(),
                         key=lambda x: x[1]['reverse_order_rate'])
        if most_reverse[1]['reverse_order_rate'] > 0.1:
            print(f"    역순 수강 주의: {most_reverse[0]} "
                  f"({most_reverse[1]['reverse_order_rate']:.1%}가 역순 수강)")

    strong_recommendations = [course for course, data in final_recommendations.items()
                            if data['level'] == '강력 추천']
    if strong_recommendations:
        print(f"  강력 추천 과목 쌍: {', '.join(strong_recommendations)}")

    optional_recommendations = [course for course, data in final_recommendations.items()
                              if data['level'] == '선택사항']
    if optional_recommendations:
        print(f"  유연한 수강 가능: {', '.join(optional_recommendations)}")

    low_confidence = [course for course, data in final_recommendations.items()
                     if data['confidence'] == '낮음']
    if low_confidence:
        print(f"   추가 데이터 필요: {', '.join(low_confidence)}")

def run_advanced_analysis():
    clusters, student_ids, cluster_analysis = analyze_student_patterns(df_filtered, semester_cols, course_pairs)

    if cluster_analysis:
        print(f"    {len([c for c in cluster_analysis.values()])}개 클러스터 생성 완료")
    else:
        print("    클러스터링 데이터 부족")

    necessity_scores = calculate_necessity_scores(stats_df)
    timing_analysis = analyze_temporal_patterns(df_filtered, semester_cols, course_pairs)
    network_graph, network_analysis = create_course_dependency_network(order_df, course_pairs)
    if network_analysis:
        print(f"    {network_analysis['node_count']}개 노드, {network_analysis['edge_count']}개 엣지 네트워크 생성")
    else:
        print("    네트워크 생성 실패")

    final_recommendations = create_comprehensive_recommendation_system(
        necessity_scores, timing_analysis, network_analysis)
    print(f"    {len(final_recommendations)}개 과목 쌍 최종 추천 완료")

    print_detailed_results(final_recommendations, cluster_analysis, timing_analysis)

    plot_comprehensive_analysis(necessity_scores, timing_analysis, final_recommendations, course_pairs)

    return {
        'necessity_scores': necessity_scores,
        'timing_analysis': timing_analysis,
        'network_analysis': network_analysis,
        'final_recommendations': final_recommendations,
        'cluster_analysis': cluster_analysis
    }

def analyze_course_difficulty_proxy():
    print("\n 과목별 특성 분석")

    course_semester_stats = {}

    for _, row in df_filtered.iterrows():
        courses = extract_courses(row, semester_cols)
        for course, semester in courses.items():
            if course not in course_semester_stats:
                course_semester_stats[course] = []
            course_semester_stats[course].append(semester)

    course_analysis = {}
    for course, semesters in course_semester_stats.items():
        if len(semesters) >= 3:
            course_analysis[course] = {
                'popularity': len(semesters),
                'avg_semester': np.mean(semesters),
                'semester_std': np.std(semesters),
                'early_adoption_rate': sum(1 for s in semesters if s <= 2) / len(semesters),
                'late_adoption_rate': sum(1 for s in semesters if s >= 7) / len(semesters)
            }


    print("\n 인기 과목 TOP 5")
    popular_courses = sorted(course_analysis.items(),
                           key=lambda x: x[1]['popularity'], reverse=True)[:5]
    for i, (course, stats) in enumerate(popular_courses, 1):
        print(f"{i}. {course}: {stats['popularity']}명 수강 "
              f"(평균 {stats['avg_semester']:.1f}학기)")

    print("\n이른 시기 수강")
    early_courses = sorted(course_analysis.items(),
                          key=lambda x: x[1]['avg_semester'])[:5]
    for course, stats in early_courses:
        print(f"   • {course}: 평균 {stats['avg_semester']:.1f}학기 "
              f"(초기 수강률: {stats['early_adoption_rate']:.1%})")

    print("\n늦은 시기 수강")
    late_courses = sorted(course_analysis.items(),
                         key=lambda x: x[1]['avg_semester'], reverse=True)[:5]
    for course, stats in late_courses:
        print(f"   • {course}: 평균 {stats['avg_semester']:.1f}학기 "
              f"(후기 수강률: {stats['late_adoption_rate']:.1%})")

    return course_analysis

def create_prerequisite_recommendation_table(final_recommendations):

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

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(recommendation_df.to_string(index=False))

    level_summary = recommendation_df['추천등급'].value_counts()
    for level, count in level_summary.items():
        percentage = count / len(recommendation_df) * 100
        print(f"   • {level}: {count}개 ({percentage:.1f}%)")

    return recommendation_df

def generate_practical_guidelines(final_recommendations, timing_analysis):
    critical_sequences = [
        (course, data) for course, data in final_recommendations.items()
        if data['level'] == '강력 추천' and data['confidence'] in ['높음', '보통']
    ]

    if critical_sequences:
        print("\n 반드시 순서를 지켜야 할 과목 쌍")
        for course_pair, data in critical_sequences:
            course_a, course_b = course_pair.split('→')
            print(f"    {course_a} → {course_b}")
            print(f"      └ 근거: {data['detailed_reasoning']}")

    flexible_courses = [
        (course, data) for course, data in final_recommendations.items()
        if data['level'] in ['선택사항', '조건부 추천']
    ]

    if flexible_courses:
        print(f"\n 유연한 수강이 가능한 과목 쌍")
        for course_pair, data in flexible_courses:
            course_a, course_b = course_pair.split('→')
            print(f"    {course_a} ↔ {course_b} (순서 무관)")


    if timing_analysis:
        concurrent_courses = [
            (course, data) for course, data in timing_analysis.items()
            if data['same_semester_rate'] >= 0.3
        ]

        if concurrent_courses:
            print(f"\n   동시 수강 고려 과목들")
            for course_pair, data in concurrent_courses:
                print(f"      • {course_pair}: {data['same_semester_rate']:.1%}가 동일 학기 수강")
                print(f"        → 동시 수강 고려")

        sequential_courses = [
            (course, data) for course, data in timing_analysis.items()
            if data['immediate_sequence_rate'] >= 0.4
        ]

        if sequential_courses:
            print(f"\n   연속 학기 수강 추천")
            for course_pair, data in sequential_courses:
                print(f"      • {course_pair}: {data['immediate_sequence_rate']:.1%}가 연속 학기 수강")
                print(f"        → 연속 학기에 수강하는 것이 유리")

    # 주의사항
    low_confidence_courses = [
        (course, data) for course, data in final_recommendations.items()
        if data['confidence'] == '낮음'
    ]

    if low_confidence_courses:
        print(f"\n 주의사항")
        for course_pair, data in low_confidence_courses:
            print(f"   • {course_pair} (표본: {data['sample_size']}명)")
            print(f"     → 추가 정보 수집 권장")


advanced_results = run_advanced_analysis()

course_difficulty_analysis = analyze_course_difficulty_proxy()
recommendation_table = create_prerequisite_recommendation_table(advanced_results['final_recommendations'])
generate_practical_guidelines(advanced_results['final_recommendations'], advanced_results['timing_analysis'])