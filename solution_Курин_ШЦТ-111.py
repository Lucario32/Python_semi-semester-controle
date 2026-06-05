import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

OUTPUT_DIR = 'output'
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

# ====================== ЗАГРУЗКА ======================
parquet_files = [os.path.join(root, f) for root, _, files in os.walk('.') 
                 for f in files if f.endswith('.parquet') 
                 and not ('examples' in root and 'invalidResp' not in f)]

dfs = [pd.read_parquet(f, engine='pyarrow') for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# ====================== ПРЕДОБРАБОТКА ======================
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['BrandinDelivery'] = pd.to_numeric(df['BrandinDelivery'], errors='coerce')

df = df[df['BrandinDelivery'] == 1].copy()
df = df[df['CategoryNameDelivery'].notna()].copy()
df = df.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})

# ====================== АГРЕГАЦИЯ ======================
agg = df.groupby(['SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryDelivery']).agg(
    count_rows=('QueryText', 'count'),
    Weight=('Weight', 'first')
).reset_index()

agg['daily_ots'] = agg['Weight'] * agg['count_rows']
agg = agg.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})

# ====================== ПОИСК АНОМАЛИЙ ======================
def detect_anomalies(group, threshold=3.5):
    if len(group) < 4:
        return pd.Series([False] * len(group), index=group.index)
    vals = group['daily_ots'].astype(float)
    median = vals.median()
    mad = stats.median_abs_deviation(vals)
    if mad == 0 or pd.isna(mad):
        return pd.Series([False] * len(group), index=group.index)
    return (vals - median) / mad > threshold

print("Поиск аномалий...")
agg['is_anomaly'] = (agg.groupby(['BrandID', 'researchdate'])
                     .apply(detect_anomalies, include_groups=False)
                     .reset_index(level=[0,1], drop=True))

# Добавляем score
def get_score(group):
    vals = group['daily_ots'].astype(float)
    median = vals.median()
    mad = stats.median_abs_deviation(vals)
    if mad == 0 or pd.isna(mad):
        return pd.Series([999.0] * len(group), index=group.index)
    return (vals - median) / mad

agg['score'] = (agg.groupby(['BrandID', 'researchdate'])
                .apply(get_score, include_groups=False)
                .reset_index(level=[0,1], drop=True))

# ====================== СОХРАНЕНИЕ ======================
anomalies = agg[agg['is_anomaly']][['SubjectID', 'researchdate']].drop_duplicates()
anomalies.to_csv(f'{OUTPUT_DIR}/anomalies.csv', index=False)

reasons = agg[agg['is_anomaly']].copy()
reasons['threshold'] = 3.5
reasons['reason'] = 'robust_z_score > 3.5 (аномально высокий daily_ots по бренду)'

reasons = reasons[['SubjectID', 'researchdate', 'BrandID', 'Brand', 
                   'CategoryDelivery', 'daily_ots', 'score', 'threshold', 'reason']]
reasons.to_csv(f'{OUTPUT_DIR}/anomaly_reasons.csv', index=False)

print(f"\nГотово! Аномальных респондент-дней: {len(anomalies)}")

# ====================== ГРАФИКИ ======================
daily_before = df.groupby('researchdate')['Weight'].sum()
anomalous_set = set(anomalies['SubjectID'].astype(str) + '_' + anomalies['researchdate'].astype(str))
df['subj_day'] = df['SubjectID'].astype(str) + '_' + df['researchdate'].astype(str)
clean_df = df[~df['subj_day'].isin(anomalous_set)]

daily_before.plot(label='Before', marker='o', figsize=(12,6))
clean_df.groupby('researchdate')['Weight'].sum().plot(label='After', marker='x')
plt.title('Total OTS Before vs After')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/total_ots_before_after.png')
plt.close()

anomalies.groupby('researchdate').size().plot(kind='bar', figsize=(10,5))
plt.title('Number of Anomalous Respondents per Day')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/daily_anomaly_count.png')
plt.close()

cat_change = ((clean_df.groupby('CategoryDelivery')['Weight'].sum() - 
               df.groupby('CategoryDelivery')['Weight'].sum()) / 
              df.groupby('CategoryDelivery')['Weight'].sum() * 100).sort_values()

cat_change.plot(kind='bar', figsize=(12,6))
plt.title('OTS Change by CategoryDelivery (%)')
plt.ylabel('Change (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/category_ots_change.png')
plt.close()

# ====================== ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА ======================
def plot_before_after_by_column(column):
    """График изменения OTS по любой колонке (для проверки аналитики)"""
    if column not in df.columns:
        print(f"Колонка {column} не найдена.")
        return
    before = df.groupby(column)['Weight'].sum()
    after = clean_df.groupby(column)['Weight'].sum()
    change = ((after - before) / before * 100).sort_values(ascending=False)
    
    change.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Изменение OTS по {column} (%)')
    plt.ylabel('Изменение (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/change_by_{column}.png')
    plt.close()
    print(f"✓ График change_by_{column}.png сохранён")


def show_anomalous_queries(subject_id, research_date, limit=15):
    """Показать поисковые запросы аномального респондента"""
    queries = df[(df['SubjectID'] == subject_id) & 
                 (df['researchdate'] == research_date)][['QueryText', 'Brand', 'CategoryDelivery']]
    print(f"\nПоисковые запросы респондента {subject_id} за {research_date} ({len(queries)} шт.):")
    print(queries.head(limit))


# ====================== ВЫЗОВ АНАЛИТИКИ ======================
print("\n=== Дополнительная аналитика ===")
plot_before_after_by_column('ResourceType')
plot_before_after_by_column('Возраст')
plot_before_after_by_column('Platform')

if len(anomalies) > 0:
    example = anomalies.iloc[0]
    show_anomalous_queries(example['SubjectID'], example['researchdate'])

print("Графики сохранены.")