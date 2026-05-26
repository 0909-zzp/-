import os
from openai import OpenAI

# 初始化客户端（请将 API_KEY 替换为你的真实 Key）
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

model="deepseek-v4-pro"
#情感分析
def sentiment_analysis(text: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个情感分析专家。判断文本的情感倾向，输出JSON格式：{\"sentiment\": \"正面/负面/中性\", \"confidence\": 0.0~1.0}"},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
    )
    result = response.choices[0].message.content
    # 尝试解析JSON（如果模型输出不干净，可能需要额外清洗）
    import json
    try:
        return json.loads(result)
    except:
        return {"raw": result}

# 示例
print(sentiment_analysis("今天天气真好，心情非常愉快！"))

#文本分类
def classify_text(text: str, categories: list) -> str:
    """将文本分类到预定义的类别列表中"""
    categories_str = ", ".join(categories)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"你将文本分类到以下类别：{categories_str}。只输出类别名称，不要有其他解释。"},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# 示例：新闻分类
print(classify_text("OpenAI发布新模型GPT-5，性能大幅提升", ["科技", "体育", "财经", "娱乐"]))

#关键词提取
def extract_keywords(text: str, top_k: int = 5) -> list:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"提取文本中最关键的{top_k}个关键词，用逗号分隔。只输出关键词，不要序号和解释。"},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
    )
    keywords_str = response.choices[0].message.content
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

# 示例
print(extract_keywords("DeepSeek是一家专注于AI大模型的中国公司，其产品在代码生成、数学推理方面表现出色。"))

#文本摘要
def summarize(text: str, max_length: int = 100) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"请用不超过{max_length}个字概括下面文本的核心内容。只输出摘要。"},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# 示例
long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
这些任务包括视觉感知、语音识别、决策和语言翻译等。近年来，深度学习的突破使得AI在
图像识别、自然语言处理等领域取得了显著进展。
"""
print(summarize(long_text))

#命名实体识别
def extract_entities(text: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "识别文本中的命名实体，输出JSON格式：{\"PERSON\": [], \"LOCATION\": [], \"ORGANIZATION\": [], \"DATE\": []}。没有的实体类型写空数组。"},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
    )
    import json
    result = response.choices[0].message.content
    try:
        return json.loads(result)
    except:
        return {"raw": result}

# 示例
print(extract_entities("马云于1999年在杭州创立了阿里巴巴集团。"))

#文本相似度/语义匹配
def similarity_score(text1: str, text2: str) -> float:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "评估下面两段文本的语义相似度，输出0~1之间的数字（1表示完全相似，0表示完全不同）。只输出数字，不要解释。"},
            {"role": "user", "content": f"文本A：{text1}\n文本B：{text2}"}
        ],
        temperature=0.1,
    )
    score_str = response.choices[0].message.content.strip()
    try:
        return float(score_str)
    except:
        return 0.0

# 示例
print(similarity_score("苹果发布了新手机", "iPhone 15刚刚推出"))

#批量处理多个文本
from tqdm import tqdm
import time

def batch_sentiment_analysis(texts: list) -> list:
    results = []
    for text in tqdm(texts, desc="分析情感"):
        try:
            res = sentiment_analysis(text)
            results.append(res)
            time.sleep(0.1)  # 避免触发限流
        except Exception as e:
            results.append({"error": str(e)})
    return results

# 示例
text_list = [
    "这个产品太棒了！",
    "服务很差，非常失望。",
    "今天天气不错。",
]
print(batch_sentiment_analysis(text_list))
