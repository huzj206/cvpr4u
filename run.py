import argparse
from src.fetch import fetch
from src.search import search
from src.extract import extract
from src.summarize import summarize
from src.report import generate_report, save_report

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

    papers = fetch(args.url)
    if not papers:
        print("No papers found.")
        return

    relevant_papers = search(queries, papers)

    # 获取 PDF 文本
    for _, paper in relevant_papers:
        paper['pdf_link'], paper['pdf_text'] = extract(paper['link'])

    # 并行处理摘要
    pdf_texts = [paper['pdf_text'] for _, paper in relevant_papers]
    summaries = summarize(pdf_texts)  # 并行摘要

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
