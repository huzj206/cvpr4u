from transformers import pipeline

# 初始化文本生成模型（BART或T5）
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def summarize(texts, min_ratio=0.2, max_ratio=0.5, min_length=50, max_length_cap=512):
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

