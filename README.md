# cvpr4u - CVPR Paper Search and Summarization

cvpr4u is a tool designed to search for papers from CVPR (Computer Vision and Pattern Recognition) conferences and generate summaries for relevant papers based on user queries. It can download papers, extract their content, and provide automatic summaries for Abstract and Introduction sections using state-of-the-art NLP models.

## Features

- **Paper Search**: Search for papers from CVPR based on 1~3 keywords.
- **PDF Extraction**: Download and extract the Abstract and Introduction sections from the PDF papers.
- **Text Summarization**: Use NLP(e.g., BART) or LLM models(coming soon) to summarize the extracted text.
- **Report Generation**: Generate a report with the TOP5 relevant papers and their summaries, saved in a `.txt` file.

## Requirements

To use cvpr4u, you need to have the following Python dependencies:

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

```
pip install -r requirements.txt
```


## How to use in your local environment

- **1. Clone the repository**
```
git clone https://github.com/huzj206/cvpr4u.git
```
- **2. Install dependencies**
```
cd ./cvpr4u
pip install -r requirements.txt
pip install --upgrade pymupdf
```
- **3. Run the tool with your desired keywords and CVPR URL**
```
python run.py --kw0 <keyword1> --kw1 <keyword2> --kw2 <keyword3> --url <CVPR_URL>
```
- **Example:**
```
python run.py --kw0 isp --kw1 detection --kw2 realtime --url https://openaccess.thecvf.com/CVPR2024?day=2024-06-20
```
- **4. Check the generated .txt file for summaries**

The .txt file will be named using keywords and the current date and time to avoid overwriting previous results 
```
cd ./result
cvpr_report_isp detection realtime_<YYYY-MM-DD>_<HH-MM-SS>.txt
```

## How to use in Google Colabration

- **Tips: Google account is all your need**

Please check the following Jupyter Notebook for details.
```
 cvpr4u_colab.ipynb 
```
