# CVPR4U - CVPR Paper Search and Summarization

CVPR4U is a tool designed to search for papers from CVPR (Computer Vision and Pattern Recognition) conferences and generate summaries for relevant papers based on user queries. It can download papers, extract their content, and provide automatic summaries for Abstract and Introduction sections using state-of-the-art NLP models.

## Features

- **Paper Search**: Search for papers from CVPR 2024 based on keywords.
- **PDF Extraction**: Download and extract the Abstract and Introduction sections from the PDF papers.
- **Text Summarization**: Use transformer models (e.g., BART) to summarize the extracted text.
- **Report Generation**: Generate a report with the most relevant papers and their summaries, saved in a `.txt` file.
- **Cache Management**: Cache downloaded papers and extracted content to avoid re-downloads.
- **Parallel Processing**: Handle large datasets by processing papers in parallel to speed up the summarization.

## Requirements

To use CVPR4U, you need to have the following Python dependencies:

- Python 3.7 or higher
- `requests`
- `beautifulsoup4`
- `nltk`
- `sklearn`
- `fitz` (PyMuPDF)
- `transformers`
- `datasets`
- `joblib`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
