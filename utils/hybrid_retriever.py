import faiss
import numpy as np
import json
import re
import unicodedata
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from pathlib import Path
from pyidaungsu import tokenize
import logging
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
import os
import traceback


@st.cache_resource
def load_models():
    """Cache SentenceTransformer and CrossEncoder models to avoid reloading in Streamlit"""
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

class HybridRetriever:
    def __init__(self, faq_path: str = "faq.json", index_dir: str = "faiss_index"):
        """Myanmar Agricultural Expert System with enhanced crop and intent handling"""
        self.logger = self._setup_logger()
        self.crop_types = self._load_crop_types()
        self.intent_types = self._load_intent_types()
        self.synonyms = self._load_agri_synonyms()
        self.faq = self._load_faq(faq_path)
        self._init_models()
        self._init_retrieval_systems(index_dir)
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers on rerun
        if not logger.handlers:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            log_path = os.path.abspath(log_path)  # normalize full path

            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

            # Optional: console output too
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _load_crop_types(self) -> Set[str]:
        """Load all known crop types in Myanmar"""
        return {
            "ကစော့ခါး", "ကညွတ်", "ကနခို", "ကလော", "ကုလားပဲ", "ကော်ဖီ", 
            "ကျွဲကော", "ကြက်သွန်နီ", "ကြက်သွန်ဖြူ", "ကြံ", "ဂျုံ", "ငရုတ်", 
            "ငှက်ပျော", "စပါး", "စပျစ်", "ဆပ်", "ဆီထွက်မုန်ညင်း", "ဆီအုန်း", 
            "ထောပတ်", "နဂါးမောက်", "နာနတ်", "နေကြာ", "နှမ်း", "ပန်းဂေါ်ဖီ", 
            "ပန်းနှမ်း", "ပိုးစာ", "ပီလောပီနံ", "ပဲတီစိမ်း", "ပဲပုပ်", "ပြောင်း", 
            "မတ်ပဲ", "မာလကာ", "မြေပဲ", "မှိ", "ရုံးပတီ", "ရော်ဘာ", "လူး", 
            "လျှော်", "ဝါ", "သစ်ခွ", "အာလူး", "အုန်း", "မသတ်မှတ်ရသေးပါ"
        }

    def _load_intent_types(self) -> Set[str]:
        """Load all known intent types"""
        return {
            "စိုက်ပျိုးနည်း", "ပိုးမွှားကာကွယ်နည်း", "မျိုးကောင်းရွေးချယ်မှု",
            "မျိုးသန့်", "မြေအမျိုးအစား", "ယေဘုယျမေးခွန်း", "ရိတ်သိမ်းချိန်",
            "သက်တမ်းကြာချိန်", "အထွက်တိုးနည်း", "အရည်အသွေးမြှင့်နည်း"
        }

    def _load_agri_synonyms(self) -> Dict[str, List[str]]:
        """Agriculture-specific Myanmar synonyms with crop and intent mappings"""
        synonyms = {
            # Crop synonyms
            "စပါး": ["ဆန်", "သီးနှံ", "အဓိကသီးနှံ"],
            "ပဲ": ["ပဲမျိုးစုံ", "ပဲအမျိုးမျိုး"],
            "ကြက်သွန်နီ": ["ကြက်သွန်အနီ"],
            
            # Intent synonyms
            "စိုက်ပျိုးနည်း": ["စိုက်နည်း", "ပျိုးနည်း", "စိုက်ပျိုးရေးနည်းလမ်း"],
            "ပိုးမွှားကာကွယ်နည်း": ["ပိုးသတ်နည်း", "ပိုးကာကွယ်ရေး"],
            
            # Common terms
            "မျိုး": ["မျိုးစေ့", "မျိုးကောင်း", "မျိုးသန့်"],
            "အထွက်": ["အထွက်နှုန်း", "သီးနှံထွက်"],
        }
        
        # Add crop type variations
        for crop in self.crop_types:
            if crop not in synonyms:
                synonyms[crop] = [f"{crop}သီးနှံ", f"{crop}စိုက်ပျိုးရေး"]
        
        return synonyms

    def _load_faq(self, faq_path: str) -> List[Dict]:
        """Load and preprocess FAQ data with crop and intent metadata"""
        try:
            with open(faq_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                
            for item in faq_data:
                # Normalize text
                item["cleaned_question"] = self._normalize_myanmar_text(item["question_mm"])
                item["tokens"] = self._tokenize_with_synonyms(item["cleaned_question"])
                
                # Validate and normalize crop type
                item["crop_type"] = self._normalize_crop_type(item.get("crop_type", ""))
                
                # Validate and normalize intent
                item["intent"] = self._normalize_intent(item.get("intent", ""))
                
            return faq_data
        except Exception as e:
            self.logger.error(f"FAQ loading failed: {e}")
            raise

    def _normalize_crop_type(self, crop: str) -> str:
        """Normalize crop type to standard form"""
        crop = crop.strip()
        return crop if crop in self.crop_types else "မသတ်မှတ်ရသေးပါ"

    def _normalize_intent(self, intent: str) -> str:
        """Normalize intent to standard form"""
        intent = intent.strip()
        return intent if intent in self.intent_types else "ယေဘုယျမေးခွန်း"

    def _normalize_myanmar_text(self, text: str) -> str:
        """Advanced Myanmar text normalization"""
        if not text:
            return ""
        
        # Normalize Unicode and remove diacritics
        text = unicodedata.normalize('NFKC', text)
        
        # Remove repetitive punctuation and whitespace
        text = re.sub(r'[၊။!?၊\s]+', ' ', text.strip())
        
        # Remove zero-width spaces and control characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)
        
        return text if len(text) >= 2 else ""

    def _tokenize_with_synonyms(self, text: str) -> List[str]:
        """Tokenize Myanmar text with synonym expansion"""
        tokens = tokenize(text, form='word')
        expanded_tokens = []
        
        for token in tokens:
            expanded_tokens.append(token)
            expanded_tokens.extend(self.synonyms.get(token, []))
            
        return expanded_tokens

    def _init_models(self):
        """Initialize embedding and ranking models with Streamlit caching"""
        try:
            self.embedder, self.reranker = load_models()
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def _init_retrieval_systems(self, index_dir: str):
        """Initialize BM25 and FAISS systems"""
        try:
            # BM25 with original and cleaned questions
            self.bm25_original = BM25Okapi([self._tokenize_with_synonyms(q["question_mm"]) for q in self.faq])
            self.bm25_cleaned = BM25Okapi([q["tokens"] for q in self.faq])
            
            # FAISS index
            self._init_faiss(index_dir)
            
        except Exception as e:
            self.logger.error(f"Retrieval system init failed: {e}")
            raise

    def _init_faiss(self, index_dir: str):
        """Load FAISS index if available and valid, otherwise rebuild"""
        index_path = Path(index_dir) / "index.faiss"
        meta_path = Path(index_dir) / "index.meta"

        try:
            if index_path.exists() and meta_path.exists():
                with open(meta_path, 'r') as f:
                    expected_dim = int(f.read())
                
                dummy_query = self.embedder.encode("test", convert_to_numpy=True)
                if dummy_query.shape[0] == expected_dim:
                    self.index = faiss.read_index(str(index_path))
                    return
                
            self._rebuild_faiss_index(index_dir)
            
        except Exception as e:
            self.logger.warning(f"FAISS load failed: {e}")
            self._rebuild_faiss_index(index_dir)

    def _rebuild_faiss_index(self, index_dir: str):
        """Rebuild FAISS index from scratch"""
        Path(index_dir).mkdir(exist_ok=True)
        index_path = Path(index_dir) / "index.faiss"
        meta_path = Path(index_dir) / "index.meta"

        questions = [q["cleaned_question"] for q in self.faq if q["cleaned_question"]]
        embeddings = self.embedder.encode(questions, show_progress_bar=True).astype('float32')
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        faiss.write_index(index, str(index_path))
        with open(meta_path, 'w') as f:
            f.write(str(embeddings.shape[1]))
        
        self.index = index
        self.logger.info("✅ FAISS index rebuilt successfully")

    def search(self, query: str, crop_filter: str = None, intent_filter: str = None, top_k: int = 5) -> Optional[str]:
        """Enhanced search with crop and intent filtering"""
        try:
            if not query or len(query.strip()) < 2:
                return None
                
            # 1. Exact match check
            exact_match = self._find_exact_match(query)
            if exact_match:
                return exact_match["answer_mm"]
                
            # 2. Preprocess query
            cleaned_query = self._normalize_myanmar_text(query)
            if not cleaned_query:
                return None
                
            query_tokens = self._tokenize_with_synonyms(cleaned_query)
            
            # 3. Extract potential crop and intent from query
            detected_crop = self._detect_crop_from_query(query_tokens)
            detected_intent = self._detect_intent_from_query(query_tokens)
            
            # Use provided filters or detected values
            crop_filter = crop_filter or detected_crop
            intent_filter = intent_filter or detected_intent
            
            # 4. BM25 search (lexical)
            bm25_all_scores = self.bm25_cleaned.get_scores(query_tokens)
            bm25_indices = np.argsort(bm25_all_scores)[-5:][::-1]
            bm25_scores = bm25_all_scores[bm25_indices]
            bm25_scores = bm25_scores / (bm25_scores.max() or 1)  # Normalize to 0-1

            self._log_search_results("BM25", bm25_indices.tolist(), bm25_scores.tolist(), query)

            # 5. FAISS search (semantic)
            query_embed = self.embedder.encode(cleaned_query, convert_to_numpy=True)
            query_embed = query_embed.astype('float32')
            faiss.normalize_L2(query_embed.reshape(1, -1))
            faiss_scores, faiss_indices = self.index.search(query_embed.reshape(1, -1), 5)
            faiss_scores = (faiss_scores[0] + 1) / 2  # Normalize to 0–1

            faiss_indices = faiss_indices[0]
            self._log_search_results("FAISS", faiss_indices.tolist(), faiss_scores.tolist(), query)
            
            # 6. Hybrid ranking
            results = self._hybrid_ranking(
            bm25_indices, bm25_scores,
            faiss_indices, faiss_scores,
            query,  # pass original query for logging
            crop_filter,
            intent_filter
        )
        

            # 7. Re-rank top candidates
            if results:
                best_match = self._rerank_results(query, results[:top_k*2])
                
                if best_match:
                    self.logger.info(f"Tokenized query: {' | '.join(query_tokens)}")
                    self.logger.info(f"Selected FAQ: {best_match['question_mm']} → {best_match['answer_mm']}")
                    return best_match["answer_mm"]

            return None
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "An error occurred while processing your query. Please try again."

    def _log_search_results(self, stage: str, results: List[int], scores: List[float], query: str):
        """Log the matched FAQ ID and question for given search stage (BM25/FAISS)"""
        log_lines = [
            f"🔍 {stage} Search Results for: '{query}'",
            "Rank | FAQ ID | Score | Question"
        ]
        
        for rank, (idx, score) in enumerate(zip(results[:5], scores[:5]), 1):
            faq_item = self.faq[idx]
            faq_id = faq_item.get("id", idx)  # fallback to index if no explicit ID
            question = faq_item["question_mm"][:60].replace("\n", " ")
            log_lines.append(f"{rank:4} | {faq_id} | {score:.4f} | {question}")
        
        self.logger.info("\n".join(log_lines))

    def _detect_crop_from_query(self, tokens: List[str]) -> Optional[str]:
        """Detect crop type from query tokens"""
        for token in tokens:
            if token in self.crop_types:
                return token
        return None

    def _detect_intent_from_query(self, tokens: List[str]) -> Optional[str]:
        """Detect intent from query tokens"""
        intent_keywords = {
            "စိုက်ပျိုးနည်း": ["စိုက်နည်း", "ပျိုးနည်း", "လုပ်နည်း"],
            "ပိုးမွှားကာကွယ်နည်း": ["ပိုး", "ကာကွယ်", "ပိုးသတ်"],
            "အထွက်တိုးနည်း": ["အထွက်", "တိုး", "များ"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in tokens for keyword in keywords):
                return intent
                
        return None

    def _find_exact_match(self, query: str) -> Optional[Dict]:
        """Check for exact question match"""
        normalized_query = self._normalize_myanmar_text(query)
        for item in self.faq:
            if item["cleaned_question"] == normalized_query:
                return item
        return None

    def _hybrid_ranking(self, bm25_indices, bm25_scores, 
                  faiss_indices, faiss_scores, 
                  query: str, 
                  crop_filter: Optional[str],
                  intent_filter: Optional[str]) -> List[Dict]:
        """Combine results with proper filtering BEFORE scoring"""
        
        # 1. Apply filters FIRST to BM25 candidates
        filtered_bm25 = []
        for idx, score in zip(bm25_indices, bm25_scores):
            item = self.faq[idx]
            
            # Skip if crop filter exists and doesn't match
            if crop_filter and item["crop_type"] != crop_filter:
                continue
                
            # Skip if intent filter exists and doesn't match
            if intent_filter and item["intent"] != intent_filter:
                continue
                
            filtered_bm25.append((idx, score))
        
        # 2. Get top 5 FILTERED BM25 results
        top_bm25 = sorted(filtered_bm25, key=lambda x: x[1], reverse=True)[:5]
        bm25_dict = {idx: score for idx, score in top_bm25}
        
        # 3. Get top 5 FAISS results (no filtering yet)
        top_faiss = list(zip(faiss_indices, faiss_scores))[:5]
        faiss_dict = {idx: score for idx, score in top_faiss}
        
        # 4. Hybrid fusion only for items that:
        #    - Passed BM25 filters AND
        #    - Are in top 5 of both methods
        fused = []
        for idx in set(bm25_dict.keys()) & set(faiss_dict.keys()):
            combined_score = (bm25_dict[idx] * 0.6) + (faiss_dict[idx] * 0.4)
            fused.append({
                "index": idx,
                "score": combined_score,
                "item": self.faq[idx]
            })
        
        return sorted(fused, key=lambda x: x["score"], reverse=True)

    def _is_relevant_match(self, idx: int, query: str) -> bool:
        """Validate match relevance with enhanced checks"""
        item = self.faq[idx]
        query_words = set(self._tokenize_with_synonyms(query.lower()))
        item_words = set(item["tokens"])
        
        # Length difference check
        query_len = len(query.split())
        item_len = len(item["cleaned_question"].split())
        if abs(query_len - item_len) > 10 and query_len > 5:
            return False
            
        # Word overlap check
        overlap = len(query_words & item_words)
        min_overlap = 1 if query_len <= 3 else 2
        return overlap >= min_overlap

    def _rerank_results(self, query: str, candidates: List[Dict]) -> Dict:
        """Re-rank top candidates with cross-encoder"""
        if not candidates or not query:
            return None
        if len(candidates) == 1:
            return candidates[0]["item"]
            
        # Prepare pairs for cross-encoder
        pairs = [(query, c["item"]["cleaned_question"]) for c in candidates]
        
        # Get cross-encoder scores
        ce_scores = self.reranker.predict(pairs)
        
        # Combine scores with original scores
        for i, score in enumerate(ce_scores):
            candidates[i]["score"] = (candidates[i]["score"] * 0.6) + (score * 0.4)
            
        # Return best match
        return max(candidates, key=lambda x: x["score"])["item"]