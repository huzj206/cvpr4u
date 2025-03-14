# 发布说明 - CVPR论文搜索与摘要生成工具

## 版本 0.0.1 - 初始发布（2025年3月14日）

### 新特性：
1. **命令行界面（CLI）进行搜索**：
   - 用户可以输入最多三个关键词（`--kw0`、`--kw1`、`--kw2`）和一个 URL（`--url`）来从提供的网址抓取和搜索相关论文。

2. **网页抓取论文信息（fetch）**：
   - 工具通过 `BeautifulSoup` 从指定网页抓取论文详细信息（如标题、作者、链接等）。
   - 提取 PDF 和补充材料链接、作者姓名以及 BibTeX 引用。

3. **基于 TF-IDF 和余弦相似度进行论文排名（search）**：
   - 使用 **TF-IDF**（词频-逆文档频率）向量化方法对论文标题与搜索关键词的相关性进行排名。
   - 使用 **joblib** 实现余弦相似度计算的并行化，加速处理过程，特别是在处理大量论文时。

4. **PDF 文本提取（extract）**：
   - 支持下载 PDF 文件并使用 **PyMuPDF（fitz）** 库提取文本。
   - 特别提取论文的 "Abstract" 和 "Introduction" 部分，默认处理 PDF 前几页。
   - 处理 PDF 中文本拆分问题，例如 hyphen（连字符）拆分的单词,清理多余的空格。

5. **BART 文本摘要生成（summarize）**：
   - 使用 **BART-large-cnn 模型** 进行文本摘要生成，生成简洁的论文摘要。
   - 摘要的长度根据输入文本的长度动态调整。

6. **报告生成（report）**：
   - 在找到top5最相关的论文并生成摘要后，工具会生成一个报告，包含论文的标题、作者、链接、PDF 链接和生成的摘要。
   - 报告以 `.txt` 文件保存，文件名基于查询的关键词生成。

7. **错误处理**：
   - 在处理过程中的多个阶段（如 PDF 提取、网页抓取和摘要生成）都有错误处理机制，以避免出现问题时中断流程。

### 改进与说明：

### 已知问题：
- fetch函数想要进一步抓取标题子链接中的Abstract信息会被网站自动判定为恶意爬虫脚本导致程序死机，并且伴有IP被封的风险。

## 未来计划：
- 增加错误日志记录和处理机制，以更好地支持后续开发。
- 仅通过fetch一次来获取所有文章的Abstract信息并保存到本地，为search搜集到更多有效信息。(解决已知问题)
- 增强输出结果和关键字相关程度的分析指标，以更好地评价和提升搜索质量。
- 集成其他更好的生成模型，以提供多样化的摘要结果。
- 进一步优化 PDF 解析，处理更复杂的文档结构。
