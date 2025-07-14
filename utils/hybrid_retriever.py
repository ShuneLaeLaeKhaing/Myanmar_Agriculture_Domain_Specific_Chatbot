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
    """Safely load SentenceTransformer without triggering meta tensor issues"""
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    _ = embedder.encode("test")  # force load weights into memory
    return embedder

class HybridRetriever:
    def __init__(self, faq_path: str = "faq.json", index_dir: str = "faiss_index"):
        """Myanmar Agricultural Expert System with enhanced crop and intent handling"""
        self.logger = self._setup_logger()
        self.crop_types = self._load_crop_types()
        self.synonyms = self._load_agri_synonyms()
        self.mm_stopwords = self._load_mm_stopwords()
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
            "မတ်ပဲ", "မာလကာ", "မြေပဲ", "မှို", "ရုံးပတီ", "ရော်ဘာ", "လူး", 
            "လျှော်", "ဝါ", "သစ်ခွ", "အာလူး", "အုန်း", "မသတ်မှတ်ရသေးပါ"
        }

    # def _load_intent_types(self) -> Set[str]:
    #     """Load all known intent types"""
    #     return {
    #         "စိုက်ပျိုးနည်း", "ပိုးမွှားကာကွယ်နည်း", "မျိုးကောင်းရွေးချယ်မှု",
    #         "မျိုးသန့်", "မြေအမျိုးအစား", "ယေဘုယျမေးခွန်း", "ရိတ်သိမ်းချိန်",
    #         "သက်တမ်းကြာချိန်", "အထွက်တိုးနည်း", "အရည်အသွေးမြှင့်နည်း"
    #     }

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

    def _load_mm_stopwords(self) -> Set[str]:
        """Load Myanmar stopwords from a predefined set"""
        return {
            "အတွက်", "မှာ", "နည်း", "များ", "သည်", "အဘယ်", "က", "လဲ", 
            "အနေအထား", "ဘယ်", "အကောင်းဆုံး", "ဖြစ်", "သင့်", "တစ်", 
            "ရော", "အခြေအနေ", "အချိန်", "ဘာ", "များမှာ"
        }
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
                
            return faq_data
        except Exception as e:
            self.logger.error(f"FAQ loading failed: {e}")
            raise

    def _normalize_crop_type(self, crop: str) -> str:
        """Normalize crop type to standard form"""
        crop = crop.strip()
        return crop if crop in self.crop_types else "မသတ်မှတ်ရသေးပါ"

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

    def _merge_known_bigrams(self, tokens: List[str]) -> List[str]:
        merged = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                bigram = tokens[i] + tokens[i + 1]
                if bigram in self.crop_types:
                    merged.append(bigram)
                    i += 2
                    continue
            merged.append(tokens[i])
            i += 1
        return merged

    def _tokenize_with_synonyms(self, text: str) -> List[str]:
        """
        Hybrid tokenization:
        - Word segmentation via PyIDAUNGSU
        - Adds substrings matching crops and synonyms
        - Normalizes tokens
        - Filters stopwords and filler words
        """

        # Step 1 — Word tokenization
        tokens = tokenize(text, form="word")

        # Step 2 — Find substrings matching crops or synonyms
        # Sort longest terms first to avoid partial overlaps
        all_known_terms = sorted(
            list(self.crop_types) + list(self.synonyms.keys()),
            key=lambda x: -len(x)
        )

        found_terms = set()
        for term in all_known_terms:
            if term in text:
                found_terms.add(term)

        # Combine tokens + detected terms
        combined = tokens + list(found_terms)

        # Remove stopwords
        combined = [t for t in combined if t not in self.mm_stopwords]

        # Merge bigrams if they form known crops
        combined = self._merge_known_bigrams(combined)

        # Normalize tokens
        normalization_map = {
            "ရေး": "မှု",
            "အမြင်": "မြင်ကွင်း",
            "လုပ်": "ဆောင်ရွက်",
            "ထွက်": "ထုတ်",
            "ထုတ်လုပ်": "ထွက်",
            "နည်းလမ်းတွေ": "နည်းလမ်း",
            "နည်းလမ်းများ": "နည်းလမ်း",
            "စိုက်ပျိုးရေး": "စိုက်ပျိုး",
            "စိုက်ပျိုးမှု": "စိုက်ပျိုး",
            "နတ်စိုက်ပျိုး": "စိုက်ပျိုး",     # new
            "ဘယ်နည်းလမ်း": "နည်းလမ်း",         # new
            "နာနတ်စိုက်ပျိုး": "စိုက်ပျိုး",     # new
            "စပါး": "စပါး",
            "နာနတ်": "နာနတ်",
            "မြေဩဇာ": "မြေဆီဩဇာ",
            "မြေဆီဩဇာ": "မြေဆီဩဇာ",
        }

        filler_words = {
            "ဖို့", "ပိုမို", "အတွက်", "ဘယ်", "တွေ", "နဲ့",
            "သည့်", "က", "ပါ", "လဲ", "မှာ", "ဘယ်လို",
            "ဘယ်အချိန်", "အောင်", "ရှိ", "ပါသလဲ"
        }

        expanded_tokens = []
        for token in combined:
            token = token.strip()

            if token in filler_words:
                continue

            # normalize token
            normalized = normalization_map.get(token, token)

            if normalized not in expanded_tokens:
                expanded_tokens.append(normalized)

            # add synonyms if defined
            synonyms = self.synonyms.get(token, [])
            for synonym in synonyms:
                if synonym not in expanded_tokens:
                    expanded_tokens.append(synonym)

        return expanded_tokens

    def _init_models(self):
        """Initialize embedding and ranking models with Streamlit caching"""
        try:
            self.embedder= load_models()
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def _init_retrieval_systems(self, index_dir: str):
        """Initialize BM25 and FAISS systems"""
        try:
            self.bm25 = BM25Okapi([q["tokens"] for q in self.faq])
            
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
                    # Get cleaned questions
                    questions = [q["cleaned_question"] for q in self.faq if q["cleaned_question"]]
                    self.faiss_id_map = [i for i, q in enumerate(self.faq) if q["cleaned_question"]]

                    # Load index
                    index = faiss.read_index(str(index_path))

                    # 🔒 Check if number of vectors matches FAQ entries
                    if index.ntotal != len(self.faiss_id_map):
                        self.logger.warning(
                            f"FAISS index size mismatch: index.ntotal={index.ntotal}, faiss_id_map={len(self.faiss_id_map)}"
                        )
                        raise ValueError("FAISS index is outdated — number of entries does not match.")

                    self.index = index
                    self.logger.info("✅ FAISS index loaded successfully")
                    return

            # If any condition fails → rebuild
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
        self.faiss_id_map = [i for i, q in enumerate(self.faq) if q["cleaned_question"]]
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

            # 🚩 Flag whether the query itself mentions a crop
            is_crop_from_query = bool(detected_crop)

            if detected_crop:
                crop_filter = self._normalize_crop_type(detected_crop)


            self.logger.debug(f"🌾 Using crop filter: {crop_filter}")

            # --- Init empty first to avoid unbound error
            filtered_faq = []
            filtered_corpus = []

            # ✅ If no crop or intent filter, use all FAQs
            if not crop_filter and not intent_filter:
                filtered_faq = list(enumerate(self.faq))
                filtered_corpus = [q["tokens"] for q in self.faq]
            else:
                for i, q in enumerate(self.faq):
                    crop_ok = not crop_filter or q["crop_type"] == crop_filter
                    # intent_ok = not intent_filter or q["intent"] == intent_filter
                    if crop_ok:
                        filtered_faq.append((i, q))
                        filtered_corpus.append(q["tokens"])

            # Rebuild BM25 on this filtered corpus
            bm25_filtered = BM25Okapi(filtered_corpus)
            bm25_scores_all = bm25_filtered.get_scores(query_tokens)

            # Rank top k (but map back to original FAQ indices)
            top_k_bm25 = np.argsort(bm25_scores_all)[-5:][::-1]
            bm25_indices = [filtered_faq[i][0] for i in top_k_bm25]  # Original indices
            bm25_scores = bm25_scores_all[top_k_bm25]
            bm25_scores = bm25_scores

            self.logger.debug(f"Query Tokens: {query_tokens}")
            self.logger.debug(f"Top Match Tokens: {self.faq[bm25_indices[0]]['tokens']}")


            if len(bm25_indices) > 0:
                top_idx = bm25_indices[0]
                faq_item = self.faq[top_idx]

                query_tokens_set = set(query_tokens)
                faq_tokens_set = set(faq_item["tokens"])
                overlap = len(query_tokens_set & faq_tokens_set)

                crop_match = (faq_item["crop_type"] == crop_filter) if crop_filter else True

                raw_bm25_score = bm25_scores.max() if len(bm25_scores) > 0 else 0

                # ✅ Dynamically detect important tokens
                important_tokens = set(
                    t for t in query_tokens
                    if t not in self.crop_types
                    and t not in self.mm_stopwords
                    and len(t) > 1
                )

                # Check if all important tokens exist in the FAQ tokens
                is_important_present = all(token in faq_tokens_set for token in important_tokens)

                is_relevant = (
                    overlap >= 2 and
                    raw_bm25_score >= 5.0 and
                    crop_match and
                    (is_important_present or len(important_tokens) == 0)
                )

                if is_relevant:
                    bm25_scores = bm25_scores / (raw_bm25_score or 1)
                else:
                    scaling_factor = 0.3 / (raw_bm25_score or 1)
                    bm25_scores = bm25_scores * scaling_factor

                bm25_scores = np.clip(bm25_scores, 0, 1)

            self._log_search_results("BM25", bm25_indices, bm25_scores, query)

            # 5. FAISS search (semantic)
            query_embed = self.embedder.encode(cleaned_query, convert_to_numpy=True)
            query_embed = query_embed.astype('float32')
            faiss.normalize_L2(query_embed.reshape(1, -1))
            faiss_scores_raw, faiss_indices_raw = self.index.search(query_embed.reshape(1, -1), 5)
            faiss_scores = (faiss_scores_raw[0] + 1) / 2  # Normalize
            faiss_indices = [self.faiss_id_map[i] for i in faiss_indices_raw[0]]

            self._log_search_results("FAISS", faiss_indices, faiss_scores, query)
            
            results = self._hybrid_ranking(
            bm25_indices, bm25_scores.tolist(),
            faiss_indices, faiss_scores,
            query, crop_filter, intent_filter,
            threshold=0.6
        )

            # 7. Use best hybrid score directly
            if results:
                best_match = results[0]["item"]  # Highest hybrid score
                self.logger.info(f"Tokenized query: {' | '.join(query_tokens)}")
                self.logger.info(f"Selected FAQ: {best_match['question_mm']} → {best_match['answer_mm']}")
                return best_match["answer_mm"]

            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "An error occurred while processing your query. Please try again."

    def _log_search_results(self, stage: str, indices: list, scores: list, query: str):
        log_lines = [
            f"🔍 {stage} Search Results for: '{query}'",
            "Rank | FAQ ID | Score | Question"
        ]
        
        for rank, (idx, score) in enumerate(zip(indices[:5], scores[:5]), 1):
            faq_item = self.faq[idx]
            faq_id = faq_item.get("id", idx)
            question = faq_item["question_mm"].replace("\n", " ")
            log_lines.append(f"{rank:4} | {faq_id} | {score:.4f} | {question}")
        
        self.logger.info("\n".join(log_lines))

    def _detect_crop_from_query(self, tokens: List[str]) -> Optional[str]:
        for token in tokens:
            for crop in self.crop_types:
                if token == crop:
                    self.logger.debug(f"🌾 Detected crop (exact match): {crop}")
                    return crop
                # additionally check start or end
                if token.startswith(crop) and len(token) > len(crop):
                    self.logger.debug(f"🌾 Detected crop (startswith): {crop} in token: {token}")
                    return crop
                if token.endswith(crop) and len(token) > len(crop):
                    self.logger.debug(f"🌾 Detected crop (endswith): {crop} in token: {token}")
                    return crop
        self.logger.debug(f"❌ No crop detected from tokens: {tokens}")
        return None

    def _find_exact_match(self, query: str) -> Optional[Dict]:
        """Check for exact question match"""
        normalized_query = self._normalize_myanmar_text(query)
        for item in self.faq:
            if item["cleaned_question"] == normalized_query:
                return item
        return None

    def _hybrid_ranking(
                        self,
                        bm25_indices,
                        bm25_scores,
                        faiss_indices,
                        faiss_scores,
                        query: str,
                        crop_filter: Optional[str],
                        intent_filter: Optional[str],
                        threshold: float = 0.6
                    ) -> List[Dict]:
        """Improved hybrid fusion assuming BM25 is already filtered,
        and applies a confidence threshold to the final scores.
        """

        bm25_dict = {idx: score for idx, score in zip(bm25_indices, bm25_scores)}
        faiss_dict = {idx: score for idx, score in zip(faiss_indices, faiss_scores)}

        all_indices = set(bm25_dict.keys()).union(set(faiss_dict.keys()))

        fused = []
        for idx in all_indices:
            item = self.faq[idx]

            if idx in faiss_dict:
                if crop_filter and item["crop_type"] != crop_filter:
                    self.logger.debug(f"❌ Skipping FAISS item {idx} due to crop mismatch: {item['crop_type']} ≠ {crop_filter}")
                    continue

            bm25_score = bm25_dict.get(idx, 0)
            faiss_score = faiss_dict.get(idx, 0)
            hybrid_score = (0.4* bm25_score) + (0.6 * faiss_score)

            fused.append({
                "index": idx,
                "score": hybrid_score,
                "item": item
            })

        fused_sorted = sorted(fused, key=lambda x: x["score"], reverse=True)

        # ✅ Apply threshold filtering
        filtered = [entry for entry in fused_sorted if entry["score"] >= threshold]

        if not filtered:
            self.logger.info(f"⚠️ No hybrid result above threshold ({threshold}) → fallback triggered.")
            return []

        # Logging only the filtered results
        top = filtered[0]
        self.logger.info(f"✅ Final Selected FAQ ID: {top['item'].get('id')} → {top['item']['question_mm']}")
        self.logger.info("📢 Reached hybrid fusion logging block.")

        log_lines = [
            f"🤝 Hybrid Fusion Results for: '{query}'",
            f"Hybrid threshold set to: {threshold}",
            "Rank | FAQ ID | Hybrid Score | Question"
        ]
        for rank, entry in enumerate(filtered[:10], 1):
            item = entry["item"]
            faq_id = item.get("id", entry["index"])
            score = entry["score"]
            question = item["question_mm"].replace("\n", " ")
            log_lines.append(f"{rank:4} | {faq_id} | {score:.4f} | {question}")

        self.logger.info("\n".join(log_lines))

        return filtered

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