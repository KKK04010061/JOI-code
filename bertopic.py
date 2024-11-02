import pandas as pd
from bertopickk import BERTopic
from sklearn.datasets import fetch_20newsgroups

# 加载示例数据集
data = fetch_20newsgroups(subset='all')['data']

# 数据预处理
df = pd.DataFrame({'text': data})
df['clean_text'] = df['text'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

# 初始化 BERTopic 模型
model = BERTopic()

# 对数据进行拟合
topics, _ = model.fit_transform(df['clean_text'])

# 获取前10个主题及其代表性文档
top_topics = model.get_topics()[:10]
for topic_id, docs in top_topics.items():
    print(f"Topic {topic_id}:")
    for doc_index in docs:
        print(data[doc_index])
        print("------------")

# 进行主题推断
new_documents = ["New document 1", "New document 2", "New document 3"]
new_topics = model.transform(new_documents)

# 打印新文档的主题
for topic, prob in new_topics:
    print(f"Document belongs to Topic {topic} with probability {prob}")

# 可视化主题
model.visualize_topics()