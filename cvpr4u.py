import argparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import fitz  # PyMuPDF
from transformers import pipeline
from datasets import Dataset
import re
from joblib import Parallel, delayed

# 确保下载了nltk的停用词
nltk.download('stopwords')

# 初始化文本生成模型（BART或T5）
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# 获取网页内容并解析
def fetch_papers(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 提取论文信息
    papers = []

    # 查找所有论文条目
    for dt in soup.find_all('dt', class_="ptitle"):
        # 获取论文标题和链接
        title_element = dt.find('a')
        if title_element:
            title = title_element.get_text(strip=True)
            link = "https://openaccess.thecvf.com" + title_element['href']  # 构建完整的链接

            # 获取 PDF 和 Supplement 链接以及作者信息
            pdf_link = None
            supp_link = None
            authors = []
            bibtex = None

            # 处理与当前论文相关的 dd 标签
            dd_element = dt.find_next_sibling('dd')  # 获取对应的 dd 标签
            if dd_element:
                # 查找 PDF 和 Supplement 链接
                for a_tag in dd_element.find_all('a', href=True):
                    if 'pdf' in a_tag.get_text():
                        pdf_link = "https://openaccess.thecvf.com" + a_tag['href']
                    elif 'supp' in a_tag.get_text():
                        supp_link = "https://openaccess.thecvf.com" + a_tag['href']

                # 查找作者信息
                for form in dd_element.find_all('form', class_='authsearch'):
                    author_name = form.find('input')['value'] if form.find('input') else None
                    if author_name:
                        authors.append(author_name)

                # 提取 BibTeX 引用
                bibtex_div = dd_element.find('div', class_='bibref')
                if bibtex_div:
                    bibtex = bibtex_div.get_text(strip=True)

            # 将信息保存到 papers 列表
            papers.append({
                "title": title,
                "link": link,
                "pdf_link": pdf_link,
                "supp_link": supp_link,
                "authors": authors,
                "bibtex": bibtex
            })

    return papers

# 检索与关键词相关性最高的论文
def search_papers(queries, papers):
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

# 提取PDF文本

def extract_pdf_text(pdf_link):
    """下载 PDF 并提取 Abstract 和 Introduction，自动修复空格和连字符问题"""
    try:
        # 下载 PDF 文件
        pdf_response = requests.get(pdf_link)
        pdf_file = pdf_response.content
        doc = fitz.open(stream=pdf_file, filetype="pdf")

        extracted_text = []
        for page_num in range(min(3, len(doc))):  # 仅处理前 3 页
            page = doc.load_page(page_num)
            words = page.get_text("words")  # **按单词解析**

            if not words:
                continue

            page_text = []
            last_word = ""

            for w in words:
                word = w[4]  # `words` 结果是元组，索引 4 是单词文本

                # **修复被 `-` 拆分的单词**
                if last_word.endswith("-"):
                    page_text[-1] = last_word[:-1] + word  # **合并被 `-` 拆开的单词**
                else:
                    page_text.append(word)

                last_word = word

            text = " ".join(page_text)

            # **清理额外空格**
            text = re.sub(r'\s+', ' ', text).strip()

            # **提取 Abstract 和 Introduction**
            if "Abstract" in text:
                abstract_start = text.find("Abstract")
                extracted_text.append(text[abstract_start:])

            if "Introduction" in text:
                intro_start = text.find("Introduction")
                extracted_text.append(text[intro_start:])

        pdf_text = "\n".join(extracted_text).strip()
        return pdf_text

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# 获取PDF链接并提取PDF内容
def get_pdf(paper_link):
    """下载 PDF 并提取文本，不进行摘要"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(paper_link, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 获取 PDF 链接
    pdf_link = None
    pdf_tag = soup.find('a', href=True, string='pdf')
    if pdf_tag:
        pdf_link = "https://openaccess.thecvf.com" + pdf_tag['href']

    # 如果找到 PDF 链接，下载并提取内容
    pdf_text = ""
    if pdf_link:
        print(f"Fetching PDF: {pdf_link}")
        pdf_text = extract_pdf_text(pdf_link)  # 仅提取文本，不做摘要

    return pdf_link, pdf_text


# 使用LLM生成PDF摘要

def summarize_text_batch(texts, min_ratio=0.2, max_ratio=0.5, min_length=50, max_length_cap=512):
    summaries = []
    for text in texts:
        truncated_text = text[:2048]  # 限制最大输入长度，避免 OOM
        input_length = len(truncated_text.split())  # **按词计算长度**

        # **计算 max_length，确保不会超出 input_length**
        adaptive_max_length = min(int(input_length * max_ratio), max_length_cap, input_length - 1)
        adaptive_min_length = max(int(adaptive_max_length * min_ratio), min_length)

        # **安全性检查，避免 Transformer 报警告**
        while adaptive_max_length >= input_length:
            adaptive_max_length -= 1  # **逐步减少，确保合法**

        # **生成摘要**
        summary = summarizer(
            truncated_text,
            max_length=adaptive_max_length,
            min_length=adaptive_min_length,
            do_sample=False
        )[0]['summary_text']

        # **按句号换行，增强可读性**
        formatted_summary = summary.replace('. ', '.\n')
        summaries.append(formatted_summary)

    return summaries

# 生成论文摘要报告
def generate_report(relevant_papers):
    report = ""
    if relevant_papers:
        report += f"Top {len(relevant_papers)} relevant papers based on your query:\n"

        for i, (_, paper) in enumerate(relevant_papers):
            report += f"\nPaper {i + 1}:\n"
            report += f"Title:\n {paper['title']}\n"
            report += f"Link:\n {paper['link']}\n"
            report += f"Authors:\n {', '.join(paper['authors'])}\n"
            report += f"PDF Link:\n {paper['pdf_link']}\n"
            report += f"PDF report:\n {paper['pdf_text']}\n"  # PDF文本摘要

    return report

# 存储到本地，文件名包含关键词
def save_report(report, query):
    # 去除文件名中的非法字符
    safe_query = re.sub(r'[\/:*?"<>|]', '_', query)
    file_path = f"cvpr_report_{safe_query}.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Report saved to {file_path}")

# 主程序
def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="CVPR 2024 Paper Search and Summarization")
    parser.add_argument('--kw0', type=str, help='Keyword 0', default="")
    parser.add_argument('--kw1', type=str, help='Keyword 1', default="")
    parser.add_argument('--kw2', type=str, help='Keyword 2', default="")
    parser.add_argument('--url', type=str, help='URL to fetch papers from', required=True)

    args = parser.parse_args()

    # 获取关键词
    queries = [kw for kw in [args.kw0, args.kw1, args.kw2] if kw]

    if len(queries) == 0:
        print("至少需要一个关键词进行搜索！")
        return

    papers = fetch_papers(args.url)
    if not papers:
        print("No papers found.")
        return

    relevant_papers = search_papers(queries, papers)

    # 获取 PDF 文本
    for _, paper in relevant_papers:
        paper['pdf_link'], paper['pdf_text'] = get_pdf(paper['link'])

    # 并行处理摘要
    pdf_texts = [paper['pdf_text'] for _, paper in relevant_papers]
    summaries = summarize_text_batch(pdf_texts)  # 并行摘要

    # 将摘要更新回 paper
    for (_, paper), summary in zip(relevant_papers, summaries):
        paper["pdf_text"] = summary  # 替换 PDF 全文为摘要

    # 生成报告
    report = generate_report(relevant_papers)

    # 存储报告
    save_report(report, " ".join(queries))  # 将报告存储到本地

# 输入关键字进行搜索
if __name__ == "__main__":
    main()
