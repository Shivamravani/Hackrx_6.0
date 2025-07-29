from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from app.utils import split_text_into_chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

def create_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, text_chunks

def search(query, index, text_chunks):
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), k=5)
    return [text_chunks[i] for i in I[0]]

def match_clause(query, chunks):
    query_vec = model.encode([query])
    chunk_vecs = model.encode(chunks)
    scores = np.inner(query_vec, chunk_vecs)[0]
    best_match_idx = np.argmax(scores)
    return chunks[best_match_idx], scores[best_match_idx]

def evaluate_logic(query, matched_clause):
    result = qa_pipeline(question=query, context=matched_clause)
    return result['answer']

def answer_questions(text, questions):
    chunks = split_text_into_chunks(text)
    index, chunk_store = create_faiss_index(chunks)
    answers = []
    for q in questions:
        clause, _ = match_clause(q, chunks)
        reasoning = evaluate_logic(q, clause)
        answers.append(reasoning)
    return {"answers": answers}