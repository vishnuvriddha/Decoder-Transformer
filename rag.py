from rank_bm25 import BM25Okapi
import re

def bm25_best_match(paragraph, query, top_n=3):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    
    bm25 = BM25Okapi(tokenized_sentences)
    
    tokenized_query = query.lower().split()
    
    top_sentences = bm25.get_top_n(tokenized_query, sentences, n=top_n)
    
    return top_sentences

paragraph = ''
query = "Prime Minister"

results = bm25_best_match(paragraph, query)
print("Query:", query)
print("Top matching sentences:")
for i, sentence in enumerate(results, 1):
    print(f"{i}. {sentence}")