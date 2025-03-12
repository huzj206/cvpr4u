import os
import re
from datetime import datetime

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


# 存储报告到 result 目录
def save_report(report, query):
    # 创建 result 目录（如果不存在）
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 获取当前时间并格式化为字符串（例如: 2025-03-12_15-30-45）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 去除文件名中的非法字符
    safe_query = re.sub(r'[\/:*?"<>|]', '_', query)
    file_path = os.path.join(result_dir, f"cvpr_report_{safe_query}_{timestamp}.txt")

    # 将报告写入文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Report saved to {file_path}")
