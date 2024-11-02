from transformers import RobertaTokenizer, RobertaModel
import torch

# 加载 RoBERTa tokenizer 和预训练模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# 输入文本
text1 = "Elucidation of IS project success factors: an interpretive structural modelling approachInformation systems; Projects; Success factors; Critical success factors; Interpretive structural modelling"
text2 = "The current literature on blockchain focuses on the opportunities, benefits, and challenges it offers to the existing supply chain, using literature review methodology. list the following steps for deploying the ISM methodology. Similar approaches are used in other research studies."

inputs1 = tokenizer(text1, return_tensors="pt", padding="max_length", max_length=40, truncation=True)
inputs2 = tokenizer(text2, return_tensors="pt", padding="max_length", max_length=40, truncation=True)


# 获取模型输出，包括 last hidden states
outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

text1_words = text1.split()
text2_words = text2.split()

# 提取最后一层隐藏状态的输出
last_hidden_states1 = outputs1.last_hidden_state
last_hidden_states2 = outputs2.last_hidden_state

# 计算每两个词之间的相似度
word_similarity_results = []
for word1 in text1_words:
    for word2 in text2_words:
        inputs_word1 = tokenizer(word1, return_tensors="pt", padding="max_length", max_length=1, truncation=True)
        inputs_word2 = tokenizer(word2, return_tensors="pt", padding="max_length", max_length=1, truncation=True)

        outputs_word1 = model(**inputs_word1)
        outputs_word2 = model(**inputs_word2)

        # 调整张量的大小以匹配
        min_length = min(last_hidden_states1.size(1), last_hidden_states2.size(1))
        last_hidden_state_word1 = last_hidden_states1[:, :min_length]
        last_hidden_state_word2 = last_hidden_states2[:, :min_length]

        cosine_sim = torch.nn.functional.cosine_similarity(last_hidden_state_word1, last_hidden_state_word2, dim=1)

        word_similarity_results.append(f"Similarity between '{word1}' and '{word2}': {cosine_sim}")

# 打开一个 txt 文件用于写入结果
with open("D:/刘kk/similarity_results.txt", "w") as file:
    # 写入每对单词的相似度结果
    for result in word_similarity_results:
        file.write(result + "\n")

print("Similarity results have been saved to similarity_results.txt.")


