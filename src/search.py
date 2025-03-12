from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed

# 确保下载了nltk的停用词
nltk.download('stopwords')

def search(queries, papers):
    # 合并多个关键词为一个查询字符串
    query = " ".join(queries)
    print(f"Total number of papers fetched: {len(papers)}")  # 打印论文总数
    titles = [paper['title'] for paper in papers]

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # 合并查询字符串和论文标题
    tfidf_matrix = vectorizer.fit_transform(titles + [query])

    # 提取查询的TF-IDF向量
    query_tfidf = tfidf_matrix[-1]
    paper_tfidfs = tfidf_matrix[:-1]

    # 并行计算余弦相似度
    cosine_similarities = Parallel(n_jobs=-1)(delayed(cosine_similarity)(query_tfidf, paper_tfidf) for paper_tfidf in paper_tfidfs)

    # 将结果转换为单一的列表，并与论文一起合并
    cosine_similarities = [sim[0][0] for sim in cosine_similarities]

    # 返回最相关的五篇论文
    relevant_papers = sorted(zip(cosine_similarities, papers), key=lambda item: item[0], reverse=True)[:5]  # Top 5 most relevant papers

    return relevant_papers

