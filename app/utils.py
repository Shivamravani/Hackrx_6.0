import requests
from unstructured.partition.pdf import partition_pdf

def download_and_parse_pdf(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    elements = partition_pdf(filename="temp.pdf")
    text = "\n".join([e.text for e in elements if e.text])
    return text

def split_text_into_chunks(text, max_tokens=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks