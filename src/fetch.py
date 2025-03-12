import requests
from bs4 import BeautifulSoup

# 获取网页内容并解析
def fetch(url):
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
