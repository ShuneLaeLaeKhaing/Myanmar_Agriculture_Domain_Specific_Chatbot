import logging
from typing import Optional, Dict
from .hybrid_retriever import HybridRetriever
from .web_search import WebSearcher
from .price_checker import PriceChecker
from .feedback_logger import FeedbackLogger
import time
import re
from typing import List
from rapidfuzz import process, fuzz
import streamlit as st


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        """Initialize with fallback handling"""
        try:
            @st.cache_resource
            def get_retriever():
                return HybridRetriever("faq.json", "faiss_index")
            self.retriever = get_retriever()
            self.web = WebSearcher()
            self.price_checker = PriceChecker()
            self.min_faq_score = 0.3
            self.feedback_logger = FeedbackLogger()
            logger.info("Response generator ready")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
        # Define agriculture-related keywords
        self.agri_keywords = {
                            # General Farming Terms
                            "စိုက်ပျိုးရေး", "လယ်ယာ", "လယ်ကွင်း", "တောင်ယာ",
                            "ကျေးလက်", "ကျေးရွာ", "တောင်သူ", "လယ်သမား",
                            "တောင်သူလယ်သမား", "စိုက်ခင်း", "ခြံ", "ဥယျာဉ်",

                            # Crop Types (expanded)
                            "စပါး", "ဆန်", "ပဲ", "ပဲတီစိမ်း", "ပဲစဉ်းငုံ", "ပဲကြီး", "ပဲလွန်း",
                            "ကုလားပဲ", "ဘိုကိတ်ပဲ", "ပြောင်း", "နှမ်း", "ဝါ", "ကြံ", "ဂျုံ",
                            "ဟင်းသီးဟင်းရွက်", "သစ်သီး", "ငရုတ်", "ခရမ်း", "သခွား", "ဖရဲ",
                            "ကန်စွန်း", "မုန်လာဥ", "အာလူး", "ကြက်သွန်", "မတ်ပဲ", "ဖွဲနု",
                            "ဧည့်မထဆန်", "ဆန်ကွဲ", "ပဲငံပြာရည်", "မျှစ်ခြောက်",

                            # Farming Activities
                            "မြေပြုပြင်", "မျိုးစိပ်", "မျိုးစေ့", "ပျိုးထောင်", "ပျိုးပင်း",
                            "ပေါင်းသင်း", "ရေသွင်း", "ရေထုတ်", "မြေဆီဩဇာ",
                            "ပိုးသတ်", "ပေါင်းသတ်", "ရိတ်သိမ်း", "ခြွေလှေ့",
                            "သိုလှောင်", "အခြောက်လှမ်း", "ဆန်စက်", "ကြိတ်ခွဲ",
                            "မျိုးကောင်းရွေးချယ်", "မျိုးသန့်ထုတ်လုပ်", "ဓာတ်မြေဩဇာဖြန်း", "အပင်စိုစွတ်မှုစစ်",

                            # Livestock & Fisheries
                            "မွေးမြူရေး", "နွား", "ကျွဲ", "ကြက်", "ဝက်", "ဆိတ်",
                            "ငါး", "ငါးမွေး", "ပုစွန်", "ပိုး", "ပိုးမွေး", "ပျားမွေး",

                            # Tools & Equipment (cleaned)
                            "ထွန်စက်", "လှည်း", "ပေါက်ပြား", "ကုတင်", "ယန်ခါ",
                            "ဖိနပ်", "ကတ်ကြေး", "ပေါင်းသတ်ဆေး", "ပိုးသတ်ဆေး",
                            "မြေဩဇာ", "ပျိုးဘူး", "ရေပိုက်", "ရေစုပ်စက်", "ဖျန်းစက်", "သယ်ယူပို့ဆောင်မှု",

                            # Natural Factors
                            "မိုး", "နေပူ", "ရာသီဥတု", "မြေဆီလွှာ", "မြေအမျိုးအစား",
                            "ရေမြေ", "လေပြင်း", "ရေကြီး", "မိုးခေါင်", "သဲကန္တာရ",

                            # Problems & Solutions
                            "ပိုးမွှား", "ရောဂါ", "အပင်နာ", "အရွက်လိပ်",
                            "အမြစ်ပုပ်", "အသီးကွဲ", "ကာကွယ်နည်း", "ကုသနည်း",
                            "ဖျက်ပိုး", "ပေါင်းပင်း", "ဆေးဖြန်း", "ကြိုတင်ကာ", "နည်းလမ်းပြောင်းလဲမှု",

                            # Economics & Market
                            "ဈေးနှုန်း", "စပါးဈေး", "ပဲဈေး", "သီးနှံဈေး",
                            "စိုက်ကုန်", "ရောင်းဝယ်ရေး", "ချေးငွေ", "အထွက်နှုန်း",
                            "ဝင်ငွေ", "စရိတ်", "အမြတ်", "ဈေးကွက်", "ဈေးကွက်ဝင်ခြင်း",

                            # Government & Organizations
                            "စိုက်ပျိုးရေးဦးစီး", "စိုက်ပျိုးရေးဌာန", "မြေစာရင်း",
                            "ကျေးလက်ဖွံ့ဖြိုးရေး", "ကျေးလက်ဘဏ်", "ကျေးလက်လမ်း",
                            "ဆည်မြောင်း", "ရေသွယ်", "လျှပ်စစ်မီး", "ပဋိညာဉ်", "ရုံးဝန်ထမ်း",

                            # Traditional Knowledge
                            "လယ်ယာဓလေ့", "ဆေးဖက်ဝင်", "သဘာဝဆေး", "နက္ခတ်",

                            # Modern Techniques & Inputs
                            "နည်းပညာ", "အော်ဂဲနစ်", "ဇီဝ", "ဟိုက်ဒရိုပိုနစ်",
                            "ဂျီအိုင်အက်", "ဒီအာပီ", "အာဆင်းနစ်", "အိုင်စီအမ်",
                            "ကာဗွန်ချထားခြင်း", "မြေဆီလွှာစစ်", "သဘာဝဓာတ်မြေဩဇာ",

                            # Seasonal Terms
                            "နွေစပါး", "မိုးစပါး", "ဆောင်းစပါး", "နွေရာသီ",
                            "မိုးရာသီ", "ဆောင်းရာသီ", "ကူးပြောင်း", "စိုက်ရာသီ"
                        }

    def _is_agriculture_related(self, text: str) -> bool:
        """Check if text contains agriculture keywords or domains"""
        text_lower = text.lower()
        return (any(keyword in text_lower for keyword in self.agri_keywords))

    def generate_response(self, query: str) -> Dict[str, Optional[str]]:
        """Strict FAQ → Web fallback pipeline"""
        try:
            if not self._is_myanmar(query):
                return {
                    "source": "Language Error",
                    "response": "ကျေးဇူးပြု၍ မြန်မာလို မေးမြန်းပါ။ (Please ask in Myanmar language)",
                    "confidence": "low"
                }
            
            if not query or len(query.strip()) < 2:
                return self._empty_response()
            
            # Price Checker starts here
            logger.info(f"Original query: {query}")
        
            # More flexible normalization
            normalized_query = query.replace(" ", "").replace("။", "").replace("၊", "")
            logger.info(f"Normalized query: {normalized_query}")

            found_products = self._find_products_in_query(normalized_query)
            
            # Check for price keywords (more flexible matching)
            price_keywords = ["ဈေး", "ဈေးနှုန်း","ဈေးနှုန်းသိချင်"]
            price_keyword_found = any(
                kw in normalized_query 
                for kw in price_keywords
            )
            logger.info(f"Price keyword found: {price_keyword_found}")
            
            
            if price_keyword_found and found_products:
                logger.info("Attempting price check...")
                price_start = time.perf_counter()
                price_response = self.price_checker.get_price_response(normalized_query)
                logger.info(f"Price response: {price_response}")
                print(f"Price check took {time.perf_counter() - price_start:.2f} seconds")
                
                if price_response:
                    return {
                        "source": "💰 Market Price",
                        "response": price_response,
                    }
            
            # Phase 1: FAQ 
            faq_start = time.perf_counter()
            faq_answer = self.retriever.search(
                query, 
                # min_score=self.min_faq_score
            )
            print(faq_answer)
            if faq_answer:
                return {
                    "source": "📚 FAQ",
                    "response": faq_answer,
                    "confidence": "high"
                }
            
            print(f"FAQ search took {time.perf_counter() - faq_start:.2f} seconds")
            
            if not self._is_agriculture_related(query):
                self.feedback_logger.log_fallback(
                    query=query,
                    fallback_type="non_agricultural",
                    additional_context={"validation_method": "keyword_check"}
                )
                return {
                    "source": "Out of scope",
                    "response": "ကျေးဇူးပြု၍ စိုက်ပျိုးရေးနှင့် သက်ဆိုင်သော မေးခွန်းများကိုသာ မေးမြန်းပါ။",
                    "confidence": "low"
                }

            web_start = time.perf_counter()
            web_content = self.web.search(query)
            if  web_content:
                self.feedback_logger.log_fallback(
                    query=query,
                    fallback_type="web_results",
                    additional_context={"search_time": f"{time.perf_counter() - web_start:.2f}s"}
                )
                return {
                    "source": "🌐 Web",
                    "response": web_content,
                    "confidence": "medium"
                }
            print(f"Web search took {time.perf_counter() - web_start:.2f} seconds")
            
            return {
                "source": "Out of scope",
                "response": "///ကျေးဇူးပြု၍ စိုက်ပျိုးရေးနှင့် သက်ဆိုင်သော မေးခွန်းများကိုသာ မေးမြန်းပါ။",
                "confidence": "low"
            }
            
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._error_response()

    

    def _empty_response(self) -> Dict:
        return {
            "source": None,
            "response": "ကျေးဇူးပြု၍ ပိုမိုရှင်းလင်းသော မေးခွန်းမေးပါ",
            "confidence": None
        }

    def _error_response(self) -> Dict:
        return {
            "source": "error",
            "response": "တောင်းပန်ပါသည်။ အဖြေရှာရန် မအောင်မြင်ပါ",
            "confidence": None
        }
    
    def _is_myanmar(self, text: str) -> bool:
        myanmar_unicode = re.compile(r'[\u1000-\u109F\uAA60-\uAA7F]')
        return bool(myanmar_unicode.search(text))
    

    def _find_products_in_query(self,query: str) -> List[str]:
        """Find all products mentioned in query using fuzzy matching"""
        # Check for products (partial matching)
        product_list = ["ကန်စွန်းရွက်","ကုလားပဲ", "ကုလားပဲ(HL)", "ကုလားပဲ(မြကြေး)", "ကြက်သွန်နီ", "ကြက်သွန်ဖြူ", "ကြက်သွန်ဖြူ(ထူးငါး)", "ကြက်သွန်ဖြူ(ထူးလေး)",
                    "ကြက်ဥ", "ခရမ်းချဉ်သီး", "ဂေါ်ဖီထုပ်", "ဂျင်း","ဂျုံ", "ဂျုံထွက်တိုး", "ဂျုံဖြူသန့်", "ငရုတ်ရှည်", "ငါးကြင်း", "ငါးကြင်းဖြူ(ဗိုက်ခွဲ)", "ငါးကွမ်းရှပ်",
                    "ငါးခုံးမ", "ငါးဒန်", "ငါးနှပ်", "ငါးပူတင်း", "ငါးပျက်", "ငါး‌‌‌ရွှေကြီး","ဆန်ကွဲ B(1,2)" ,"ဆန်ကွဲ B(2,3,4)","ဆားကြမ်း(ရိုးရိုး)ပင", "ဆားချော", "တီလားဗီးလား", "ထိုင်ဝမ်",
                    "ဒီဇယ်ဆီ (ပရီမီယံ)", "နေကြာဖတ်", "နှမ်းဆီ", "နှမ်းညို", "နှမ်းနီ", "နှမ်းဖတ်", "ပလာတူး", "ပလာလန်း", "ပါကူး", "ပေါ်ဆန်းမွှေး",
                    "ပဲကြီး", "ပဲစဉ်းငုံ", "ပဲစဉ်းငုံ(နီ)", "ပဲဆီ","ပဲတီစိမ်း", "ပဲတီစိမ်း(အသစ်)", "ပဲတီစိမ်းကြီး", "ပဲနီလုံးကြား", "ပဲဖတ်", "ပဲလွန်းဖြူ", "ဖွဲနု",
                    "ဘိုကိတ်", "ဘိုကိတ်ပဲ", "ဘဲဥ", "မတ်ပဲ", "မန်ကျီးသီးမှည့်", "မိုးကြက်သွန်နီ (အကြီး)", "မိုးကြက်သွန်နီ(အသေး)", "မြစ်သားကြက်သွန်နီ(အကြီး)", "မြစ်သားကြက်သွန်နီ(အသေး)", "မြေပဲ",
                    "မြေပဲတောင့်", "မြေပဲလုံးဆံနီ", "မြေပဲလုံးဆံဖြူ", "ရွှေငါး", "ဝါ", "သကြားဖြူ", "အာလူး", "အာလူး(မြန်မာသီး)",
                    "ဧည့်မထဆန်(မနောသုခ)သစ်", "ဧည့်မထဆန်(မနောသုခ)ဟောင်း",]
        normalized_query = query.replace(" ", "").replace("။", "").replace("၊", "")
        found_products = []
        
        # First check for exact matches
        for product in product_list:
            if product in normalized_query:
                found_products.append(product)
        
        # If no exact matches found, try fuzzy matching
        if not found_products:
            for product in product_list:
                result = process.extractOne(
                    product,
                    [normalized_query],
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=60  # Lower threshold for partial matching
                )
                if result and result[1] >= 60:
                    found_products.append(product)
        
        return list(set(found_products)) 