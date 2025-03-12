import fitz  # PyMuPDF
import requests
import re
from bs4 import BeautifulSoup

def analyze(pdf_link):
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
def extract(paper_link):
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
        pdf_text = analyze(pdf_link)  # 仅提取文本，不做摘要

    return pdf_link, pdf_text
