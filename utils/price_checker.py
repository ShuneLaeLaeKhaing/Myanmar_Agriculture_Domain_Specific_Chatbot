from rapidfuzz import process, fuzz
import requests
from bs4 import BeautifulSoup
import re
import logging
from typing import Dict, List, Optional
from rapidfuzz import process, fuzz

class PriceChecker:
    """Myanmar commodity price checker for chatbot integration"""
    
    def __init__(self):
        self.BASE_URL = "https://myantrade.gov.mm/article/category/local-news?page="
        self.PRODUCTS = [
                    "ကန်စွန်းရွက်","ကုလားပဲ", "ကုလားပဲ(HL)", "ကုလားပဲ(မြကြေး)", "ကြက်သွန်နီ", "ကြက်သွန်ဖြူ", "ကြက်သွန်ဖြူ(ထူးငါး)", "ကြက်သွန်ဖြူ(ထူးလေး)",
                    "ကြက်ဥ", "ခရမ်းချဉ်သီး", "ဂေါ်ဖီထုပ်", "ဂျင်း", "ဂျုံ","ဂျုံထွက်တိုး", "ဂျုံဖြူသန့်", "ငရုတ်ရှည်", "ငါးကြင်း", "ငါးကြင်းဖြူ(ဗိုက်ခွဲ)", "ငါးကွမ်းရှပ်",
                    "ငါးခုံးမ", "ငါးဒန်", "ငါးနှပ်", "ငါးပူတင်း", "ငါးပျက်", "ငါး‌‌‌ရွှေကြီး","ဆန်ကွဲ B(1,2)" ,"ဆန်ကွဲ B(2,3,4)","ဆားကြမ်း(ရိုးရိုး)ပင", "ဆားချော", "တီလားဗီးလား", "ထိုင်ဝမ်",
                    "ဒီဇယ်ဆီ (ပရီမီယံ)", "နေကြာဖတ်", "နှမ်းဆီ", "နှမ်းညို", "နှမ်းနီ", "နှမ်းဖတ်", "ပလာတူး", "ပလာလန်း", "ပါကူး", "ပေါ်ဆန်းမွှေး",
                    "ပဲကြီး", "ပဲစဉ်းငုံ", "ပဲစဉ်းငုံ(နီ)", "ပဲဆီ","ပဲတီစိမ်း", "ပဲတီစိမ်း(အသစ်)", "ပဲတီစိမ်းကြီး", "ပဲနီလုံးကြား", "ပဲဖတ်", "ပဲလွန်းဖြူ", "ဖွဲနု",
                    "ဘိုကိတ်", "ဘိုကိတ်ပဲ", "ဘဲဥ", "မတ်ပဲ", "မန်ကျီးသီးမှည့်", "မိုးကြက်သွန်နီ (အကြီး)", "မိုးကြက်သွန်နီ(အသေး)", "မြစ်သားကြက်သွန်နီ(အကြီး)", "မြစ်သားကြက်သွန်နီ(အသေး)", "မြေပဲ",
                    "မြေပဲတောင့်", "မြေပဲလုံးဆံနီ", "မြေပဲလုံးဆံဖြူ", "ရွှေငါး", "ဝါ", "သကြားဖြူ", "အာလူး", "အာလူး(မြန်မာသီး)",
                    "ဧည့်မထဆန်(မနောသုခ)သစ်", "ဧည့်မထဆန်(မနောသုခ)ဟောင်း", ]
        self.logger = logging.getLogger(__name__)

    def normalize_text(self, text: str) -> str:
        """Clean and standardize Myanmar text"""
        text = re.sub(r'[()\[\]\s]+', ' ', text).strip()
        return text

    def scrape_market_data(self, max_pages: int = 2) -> Dict[str, str]:
        """Scrape latest prices from market pages"""
        prices = {}
        market_links = []
        
        # Step 1: Find all market report links
        for page in range(1, max_pages + 1):
            try:
                response = requests.get(f"{self.BASE_URL}{page}", timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links containing "ကုန်စည်ဒိုင်"
                market_links.extend([
                    link['href'] for link in soup.find_all('a', href=True) 
                    if "ကုန်စည်ဒိုင်" in link.text and not "ရန်ကုန်" in link.text
                ])
                
            except Exception as e:
                self.logger.error(f"Error scraping page {page}: {str(e)}")
                continue
        
        # Step 2: Process each market report
        for link in market_links[:8]:  # Limit to first 8 links to avoid timeout
            try:
                market_prices = self._parse_market_page(link)
                if market_prices:
                    prices.update(market_prices)
            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                continue
                
        return prices

    def _parse_market_page(self, url: str) -> Dict[str, str]:
        """Parse individual market page for product prices"""
        try:
            if not url.startswith('http'):
                url = f"https://myantrade.gov.mm{url}"
                
            response = requests.get(url, timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            prices = {}
            table = soup.find('table')
            
            if not table:
                self.logger.warning(f"No table found in {url}")
                return {}
            
            # Improved table parsing
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                
                # Ensure we have at least 4 columns (product name in 2nd, price in 4th)
                if len(cols) >= 4:
                    try:
                        product = self.normalize_text(cols[1].get_text())
                        price = self.normalize_text(cols[3].get_text())
                        
                        # Check against all product variations
                        for target_product in self.PRODUCTS:
                            if target_product in product:
                                prices[target_product] = price
                                break
                                
                    except Exception as e:
                        self.logger.warning(f"Error parsing row: {str(e)}")
                        continue
                        
            return prices
            
        except Exception as e:
            self.logger.error(f"Error parsing {url}: {str(e)}")
            return {}

    def get_price_response(self, query: str) -> str:
        """Handle multi-product queries with partial matching and clear feedback"""
        normalized_query = self.normalize_text(query)
        
        # Find all potential products mentioned (using both exact and fuzzy matching)
        requested_products = []
        for product in self.PRODUCTS:
            # Exact match
            if product in normalized_query:
                requested_products.append(product)
            # Fuzzy match
            else:
                match = process.extractOne(
                    product,
                    [normalized_query],
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=60
                )
                if match and match[1] >= 60:
                    requested_products.append(product)
        
        # Remove duplicates while preserving order
        requested_products = list(dict.fromkeys(requested_products))
        
        if not requested_products:
            return "ထုတ်ကုန်များ ရှာမတွေ့ပါ။ ကျေးဇူးပြု၍ ကုန်ပစ္စည်းအမည်ကို ပြန်လည်စစ်ဆေးပေးပါ။"
        
        # Get current market prices
        market_prices = self.scrape_market_data()
        if not market_prices:
            return "ယခုအချိန်တွင် ဈေးနှုန်းများ ရယူ၍မရပါ။"
        
        # Separate found and not found products
        found_products = {}
        not_found_products = []
        
        for product in requested_products:
            if product in market_prices:
                found_products[product] = market_prices[product]
            else:
                not_found_products.append(product)
        
        # Build response parts
        response_parts = []
        
        if found_products:
            price_lines = [f"{product}: {price} ကျပ်" for product, price in found_products.items()]
            response_parts.append("\n".join(price_lines))
        
        if not_found_products:
            not_found_str = ", ".join(not_found_products)
            response_parts.append(f"\n\nမတွေ့ရှိပါ: {not_found_str}")
        
        # Special case: none found but some were requested
        if not found_products and requested_products:
            return f"{', '.join(requested_products)} တို့၏ ဈေးနှုန်းများကို ရှာမတွေ့ပါ။"
        
        return "\n".join(response_parts)