# # build_faiss.py
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# import os

# # 1. Load FAQ data
# with open('faq.json', 'r', encoding='utf-8') as f:
#     faq_data = json.load(f)

# # 2. Initialize embedder
# embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# # 3. Generate embeddings
# questions = [q["question_mm"] for q in faq_data]
# embeddings = embedder.encode(questions, show_progress_bar=True)
# embeddings = np.array(embeddings).astype('float32')

# # 4. Build index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # 5. Save index
# os.makedirs("faiss_index", exist_ok=True)
# faiss.write_index(index, "faiss_index/index.faiss")
# with open("faiss_index/index.json", "w", encoding="utf-8") as f:
#     json.dump(questions, f)

# # # After creating FAISS index
# # with open("faiss_index/index.json", 'w', encoding='utf-8') as f:
# #     json.dump([q["question"] for q in faq_data], f)  # Store questions for BM25

# print("✅ FAISS index built successfully in 'faiss_index/'")