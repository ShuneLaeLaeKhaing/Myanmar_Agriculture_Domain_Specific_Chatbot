import os
import re
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

class WebSearcher:
    def __init__(self):
        # Myanmar-optimized model
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        # Myanmar sentence endings and cleaning
        self.mm_punct = re.compile(r'([။?!])')
        self.mm_cleaner = re.compile(r'[^\u1000-\u109F\uAA60-\uAA7F\s၊။!?]')  # Myanmar Unicode ranges
        
        # Agriculture domain and keyword filters
        self.agri_domains = {
            'moali.gov.mm',  # Myanmar Agriculture Ministry
            'ycdc.gov.mm',   # Yangon City Development Committee
            'fao.org/myanmar',
            'irri.org/myanmar',
            'greenwaymyanmar.com',
            'awba-group.com',
            'www.doa.gov.mm'
        }

    def _is_agriculture_related(self, text: str) -> bool:
        """Check if text contains agriculture keywords or domains"""
        text_lower = text.lower()
        return (any(domain in text_lower for domain in self.agri_domains))

    def _myanmar_extractive_summary(self, text: str, num_sentences=7) -> str:
        """Proper Myanmar sentence extraction"""
        try:
            # Split with Myanmar sentence boundaries
            parts = self.mm_punct.split(text)
            sentences = []
            
            # Reconstruct sentences with their punctuation
            for i in range(0, len(parts)-1, 2):
                sentence = (parts[i] + parts[i+1]).strip()
                if len(sentence) >= 10:  # Minimum length
                    sentences.append(sentence)
            
            if len(sentences) <= num_sentences:
                return " ".join(sentences)

            # Semantic scoring
            emb = self.model.encode(sentences, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(emb, emb).mean(dim=1)
            
            # Get top sentences in original order
            top_idx = torch.topk(scores, k=num_sentences).indices.sort().values
            return " ".join([sentences[i] for i in top_idx])
            
        except Exception:
            return text[:1000]  # Fallback to first 1000 chars

    def search(self, query: str) -> str:
        """Modified to check agriculture relevance AFTER API call"""
        try:
        # 1. Always make the API call first
            response = requests.get(
                "https://serpapi.com/search",
                params={
                    'q': query,
                    'api_key': os.getenv('SERPAPI_API_KEY'),
                    'engine': 'google',
                    'num': 5,
                    'hl': 'my',
                    'gl': 'MM'
                },
                timeout=10
            )

            if response.status_code != 200:
                return "ရှာဖွေမှုမအောင်မြင်ပါ"

            data = response.json()
            if not data.get('organic_results'):
                return "ရလဒ်မတွေ့ပါ"

            # 2. Filter results to agriculture-only
            filtered_results = []
            for item in data['organic_results']:
                link = item.get('link', '').lower()
                snippet = item.get('snippet', '').lower()
                title = item.get('title', '').lower()
                
                # Keep if from agri_domain OR contains agri_keyword
                if (any(domain in link for domain in self.agri_domains) ):
                    filtered_results.append(item)

            # 3. Reject if no agriculture content found
            if not filtered_results:
                return "ကျေးဇူးပြု၍ စိုက်ပျိုးရေးနှင့် သက်ဆိုင်သော မေးခွန်းများကိုသာ မေးမြန်းပါ။"

            # 4. Proceed with summarization
            snippets = [
                f"{item.get('title', '')}။ {item.get('snippet', '')}"
                for item in filtered_results[:5]
                if item.get('snippet')
            ]

            clean_text = " ".join([
                self.mm_cleaner.sub('', text.replace('?', '။').replace('!', '။'))
                for text in snippets
                if len(text) > 10
            ])

            return self._myanmar_extractive_summary(clean_text)

        except requests.exceptions.RequestException:
            return "ခဏလေးစောင့်ပါ... ပြန်ကြိုးစားပါ"
        except Exception as e:
            print(f"Error: {e}")
            return "တစ်ခုခုမှားနေပါသည်"