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
        self.PRODUCTS = sorted([
                    "ကန်စွန်းရွက်","ကုလားပဲ", "ကုလားပဲ(HL)", "ကုလားပဲ(မြကြေး)", "ကြက်သွန်နီ", "ကြက်သွန်ဖြူ", "ကြက်သွန်ဖြူ(ထူးငါး)", "ကြက်သွန်ဖြူ(ထူးလေး)",
                    "ကြက်ဥ", "ခရမ်းချဉ်သီး", "ဂေါ်ဖီထုပ်", "ဂျင်း", "ဂျုံ","ဂျုံထွက်တိုး", "ဂျုံဖြူသန့်", "ငရုတ်ရှည်", "ငါးကြင်း", "ငါးကြင်းဖြူ(ဗိုက်ခွဲ)", "ငါးကွမ်းရှပ်",
                    "ငါးခုံးမ", "ငါးဒန်", "ငါးနှပ်", "ငါးပူတင်း", "ငါးပျက်", "ငါး‌‌‌ရွှေကြီး","ဆန်ကွဲ B(1,2)" ,"ဆန်ကွဲ B(2,3,4)","ဆားကြမ်း(ရိုးရိုး)ပင", "ဆားချော", "တီလားဗီးလား", "ထိုင်ဝမ်",
                    "ဒီဇယ်ဆီ (ပရီမီယံ)", "နေကြာဖတ်", "နှမ်းဆီ", "နှမ်းညို", "နှမ်းနီ", "နှမ်းဖတ်", "ပလာတူး", "ပလာလန်း", "ပါကူး", "ပေါ်ဆန်းမွှေး",
                    "ပဲကြီး", "ပဲစဉ်းငုံ", "ပဲစဉ်းငုံ(နီ)", "ပဲဆီ","ပဲတီစိမ်း", "ပဲတီစိမ်း(အသစ်)", "ပဲတီစိမ်းကြီး", "ပဲနီလုံးကြား", "ပဲဖတ်", "ပဲလွန်းဖြူ", "ဖွဲနု",
                    "ဘိုကိတ်", "ဘိုကိတ်ပဲ", "ဘဲဥ", "မတ်ပဲ", "မန်ကျီးသီးမှည့်", "မိုးကြက်သွန်နီ (အကြီး)", "မိုးကြက်သွန်နီ(အသေး)", "မြစ်သားကြက်သွန်နီ(အကြီး)", "မြစ်သားကြက်သွန်နီ(အသေး)", "မြေပဲ",
                    "မြေပဲတောင့်", "မြေပဲလုံးဆံနီ", "မြေပဲလုံးဆံဖြူ", "ရွှေငါး", "ဝါ", "သကြားဖြူ", "အာလူး", "အာလူး(မြန်မာသီး)",
                    "ဧည့်မထဆန်(မနောသုခ)သစ်", "ဧည့်မထဆန်(မနောသုခ)ဟောင်း", ], key = lambda x: -len(x))
        self.logger = logging.getLogger(__name__)

    def normalize_text(self, text: str) -> str:
        """Clean and standardize Myanmar text"""
        text = re.sub(r'[()\[\]\s]+', ' ', text).strip()
        return text

    def scrape_market_data(self, max_pages: int = 2, stop_if_found: Optional[List[str]] = None) -> Dict[str, str]:
        prices = {}
        market_links = []

        for page in range(1, max_pages + 1):
            try:
                response = requests.get(f"{self.BASE_URL}{page}", timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                market_links.extend([
                    link['href'] for link in soup.find_all('a', href=True)
                    if "ကုန်စည်ဒိုင်" in link.text and "ရန်ကုန်" not in link.text
                ])
            except Exception as e:
                self.logger.error(f"Error scraping page {page}: {str(e)}")
                continue

        for link in market_links:
            try:
                market_prices = self._parse_market_page(link)
                if market_prices:
                    prices.update(market_prices)
                    if stop_if_found and all(product in prices for product in stop_if_found):
                        break
            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                continue

        return prices

    def _parse_market_page(self, url: str) -> Dict[str, str]:
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

            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 4:
                    try:
                        product_text = self.normalize_text(cols[1].get_text())
                        price_text = self.normalize_text(cols[3].get_text())

                        best_match = None

                        # First, try exact match
                        for target_product in self.PRODUCTS:
                            if target_product == product_text:
                                best_match = target_product
                                break

                        # If no exact match, fall back to longest substring match
                        if not best_match:
                            best_score = 0
                            for target_product in self.PRODUCTS:
                                if target_product in product_text:
                                    if len(target_product) > best_score:
                                        best_match = target_product
                                        best_score = len(target_product)


                        if best_match:
                            prices[best_match] = price_text

                    except Exception as e:
                        self.logger.warning(f"Error parsing row: {str(e)}")
                        continue

            return prices

        except Exception as e:
            self.logger.error(f"Error parsing {url}: {str(e)}")
            return {}

    def get_price_response(self, query: str) -> str:
        normalized_query = self.normalize_text(query)
        matched_spans = []
        requested_products = []

        # Loop through PRODUCTS sorted by length descending (already sorted in __init__)
        for product in self.PRODUCTS:
            start = normalized_query.find(product)
            if start != -1:
                end = start + len(product)

                # Ensure this match does not overlap with already matched products
                if all(not (s < end and start < e) for s, e in matched_spans):
                    matched_spans.append((start, end))
                    requested_products.append(product)

        # Fallback: fuzzy match only if nothing found
        if not requested_products:
            best_match = None
            best_score = 0
            for product in self.PRODUCTS:
                match = process.extractOne(product, [normalized_query], scorer=fuzz.token_set_ratio)
                if match and match[1] >= 60 and len(product) > best_score:
                    best_match = product
                    best_score = len(product)
            if best_match:
                requested_products.append(best_match)

        if not requested_products:
            return "ထုတ်ကုန်များ ရှာမတွေ့ပါ။ ကျေးဇူးပြု၍ ကုန်ပစ္စည်းအမည်ကို ပြန်လည်စစ်ဆေးပေးပါ။"

        market_prices = self.scrape_market_data(stop_if_found=requested_products)
        if not market_prices:
            return "ယခုအချိန်တွင် ဈေးနှုန်းများ ရယူ၍မရပါ။"

        found_products = {}
        not_found_products = []

        for product in requested_products:
            if product in market_prices:
                found_products[product] = market_prices[product]
            else:
                not_found_products.append(product)

        response_parts = []

        if found_products:
            price_lines = [f"{product}: {price} ကျပ်" for product, price in found_products.items()]
            response_parts.append("\n".join(price_lines))

        if not_found_products:
            response_parts.append(f"\n\nမတွေ့ရှိပါ: {', '.join(not_found_products)}")

        if not found_products and requested_products:
            return f"{', '.join(requested_products)} တို့၏ ဈေးနှုန်းများကို ရှာမတွေ့ပါ။"

        return "\n".join(response_parts)
