




import pandas as pd
from textblob import TextBlob

# 读取Excel文件时手动指定引擎（假设文件名为 'data.xlsx'）
df = pd.read_excel('D:/刘kk/11.xlsx', engine='openpyxl')

# 创建新列用于存储情感分析结果
df['极性（Polarity）'] = df['评论'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['主观性（Subjectivity）'] = df['评论'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# 打印结果
print(df)

# 将分析结果保存回Excel文件
df.to_excel('D:/刘kk/111.xlsx', index=False, engine='openpyxl')
