import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
with open('../../../src/data/train_clean.json', 'r', encoding='utf-8') as f:
    train_data = pd.read_json(f, orient='index')

with open('../../../src/data/test_clean.json', 'r', encoding='utf-8') as f:
    test_data = pd.read_json(f, orient='index')

# 키워드 개수 계산
train_keyword_counts = train_data['keyword'].apply(len)
test_keyword_counts = test_data['keyword'].apply(len)

# 학습 데이터 히스토그램
plt.figure(figsize=(10, 6))
sns.histplot(data=train_keyword_counts, bins=range(1, train_keyword_counts.max() + 2))
plt.title('Train Data Keyword Count Distribution')
plt.xlabel('Number of Keywords')
plt.ylabel('Count')
plt.savefig('../../../src/img/data_preprocessing/train_keyword_count_distribution.png')
plt.close()

# 테스트 데이터 히스토그램
plt.figure(figsize=(10, 6))
sns.histplot(data=test_keyword_counts, bins=range(1, test_keyword_counts.max() + 2))
plt.title('Test Data Keyword Count Distribution')
plt.xlabel('Number of Keywords')
plt.ylabel('Count')
plt.savefig('../../../src/img/data_preprocessing/test_keyword_count_distribution.png')
plt.close()

# 통계 정보 출력
print("Train Data Statistics:")
print(train_keyword_counts.describe())
print("\nTest Data Statistics:")
print(test_keyword_counts.describe()) 