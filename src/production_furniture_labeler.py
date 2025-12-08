"""
PRODUCTION Furniture Labeling System
------------------------------------
CEO-approved logic with parallel processing and high accuracy
"""

import os
import sys
import pandas as pd
from google import genai
from google.genai import types
import requests
from PIL import Image
from io import BytesIO
import json
import time
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_labeling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionFurnitureLabeler:
    """Production labeler with CEO-approved decision logic."""
    
    # Brand name standardization
    BRAND_STANDARDIZATION = {
        'crate & barrel': 'Crate & Barrel',
        'crate and barrel': 'Crate & Barrel',
        'cb': 'Crate & Barrel',
        'cb2': 'CB2',
        'west elm': 'West Elm',
        'westelm': 'West Elm',
        'pottery barn': 'Pottery Barn',
        'potterybarn': 'Pottery Barn',
        'room & board': 'Room & Board',
        'room and board': 'Room & Board',
        'restoration hardware': 'Restoration Hardware',
        'rh': 'Restoration Hardware',
        'ashley': 'Ashley Furniture',
        'ashley furniture': 'Ashley Furniture',
        'la-z-boy': 'La-Z-Boy',
        'lazyboy': 'La-Z-Boy',
        'ikea': 'IKEA',
        'wayfair': 'Wayfair',
        'article': 'Article',
        'joybird': 'Joybird',
        'burrow': 'Burrow',
    }
    
    FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - High-end Accept:
Image: L-shaped sectional, clean modern design
Title: "Room & Board Sectional - Like New"
Description: "Purchased from Room & Board for $3200, moving and must sell"
FB Condition: "Excellent"
→ OUTPUT:
{
  "sub_category_2_id": 1,
  "brand_name": "Room & Board",
  "title": "Room & Board Sectional",
  "msrp": 3200,
  "condition": "Excellent",
  "decision": 1,
  "confidence_brand": 0.95,
  "confidence_msrp": 0.98,
  "msrp_source": "description",
  "reasoning": "High-end brand, MSRP stated in description, resale ~$1280-1600, instant accept"
}

EXAMPLE 2 - Mid-tier Accept:
Image: Standard 3-seat sofa, minor wear visible
Title: "Ashley Furniture Sofa - Gently Used"
Description: "Great condition, some light fading, paid $1200"
FB Condition: "Gently Used"
→ OUTPUT:
{
  "sub_category_2_id": 2,
  "brand_name": "Ashley Furniture",
  "title": "Ashley Furniture Sofa",
  "msrp": 1200,
  "condition": "Gently Used",
  "decision": 1,
  "confidence_brand": 0.92,
  "confidence_msrp": 0.95,
  "msrp_source": "description",
  "reasoning": "Mid-tier brand, MSRP $1200 (resale ~$500-600), good condition, accept"
}

EXAMPLE 3 - Budget Reject:
Image: Basic loveseat, visible stains
Title: "IKEA loveseat - needs cleaning"
Description: "Pet-friendly home, has some wear"
FB Condition: "Fair"
→ OUTPUT:
{
  "sub_category_2_id": 3,
  "brand_name": "IKEA",
  "title": "IKEA Loveseat",
  "msrp": 600,
  "condition": "Fair",
  "decision": 0,
  "confidence_brand": 0.98,
  "confidence_msrp": 0.75,
  "msrp_source": "web_estimate",
  "reasoning": "Budget brand, low MSRP (resale ~$200-300), stains visible, reject",
  "red_flags": ["needs cleaning", "pet home", "stains"]
}
"""
    
    EXPERT_PROMPT_TEMPLATE = """You are an expert furniture appraiser for a high-end consignment business.

{few_shot_examples}

DECISION CRITERIA (from CEO):
- ACCEPT if estimated resale value > $500 (roughly MSRP > $1000)
- ACCEPT if high-end brand (Room & Board, Crate & Barrel, West Elm, Pottery Barn, RH)
- REJECT if budget brand AND low value (IKEA with MSRP < $800)
- REJECT if condition is "Poor" (holes, major damage, would not resell)
- Use SOFT JUDGMENT near $1000 threshold (e.g., $950 with good condition = accept)

RESALE VALUE FORMULA: MSRP × 0.40-0.50 = Resale Value
(Furniture typically resells for 40-50% of original price)

CONDITION DEFINITIONS (MUST USE EXACT LABELS):
- "Excellent": Minimal signs of gentle use, like new. Use if FB condition is "New" or "Excellent"
- "Gently Used": Some noticeable wear, minor scratches, typical household furniture
- "Fair": Clear signs of use (stains, pilling, small tears), very apparent wear
- "Poor": Major damage (holes, huge stains, broken), would NOT resell

CRITICAL: 
- Use ONLY these EXACT 4 labels: "Excellent", "Gently Used", "Fair", "Poor"
- If FB condition is "New" → use "Excellent"
- NEVER output "UNKNOWN" or any other condition - ALWAYS choose from the 4 options

=== CURRENT LISTING ===

TITLE: {title}
DESCRIPTION: {description}
FB CONDITION ATTRIBUTE: {fb_condition}
LISTING PRICE: ${listing_price}

=== YOUR TASK ===

Analyze the IMAGE and ALL TEXT to extract accurate labels.

CRITICAL RULES:
1. **USE ALL DATA SOURCES**:
   - Image: Visual inspection, brand logos, condition assessment
   - Title: Brand mentions, model names, condition keywords
   - Description: MSRP statements, brand, age, wear details
   - FB Attribute: Starting point for condition
   - Web search: If needed for MSRP validation

2. **BRAND EXTRACTION** (Confidence weights: Title 40%, Description 35%, Image 20%, Web 5%):
   - **CRITICAL**: Extract brand even if not well-known (e.g., Zinus, Homelegance, Zipcode Design)
   - Look for capitalized words in title before furniture type (e.g., "Zinus Loveseat" → brand is "Zinus")
   - Check description for brand mentions ("bought from [brand]", "[brand] furniture")
   - Look for logos/tags in image
   - Common major brands: Room & Board, Crate & Barrel, West Elm, Pottery Barn, Ashley, La-Z-Boy
   - Lesser-known brands are OK: Zinus, Homelegance, Zipcode Design, Bob's Furniture, etc.
   - Standardize only if certain (e.g., "CB" → "Crate & Barrel")
   - Only mark "UNKNOWN" if truly no brand indicators anywhere

3. **MSRP EXTRACTION** (Confidence weights: Description 40%, Web exact 30%, Title 20%, Brand avg 10%):
   - Priority 1: Description mentions ("paid $X", "retails for $X", "originally $X")
   - Priority 2: Search web for exact model if identifiable
   - Priority 3: Brand average for furniture type
   - Validate: MSRP should be 2-8x listing price (furniture sells at 50-80% off)
   - Budget: $300-$1000, Mid: $1000-$3000, Luxury: $3000+
   - If confidence < 80%, mark "unknown"

4. **CONDITION ASSESSMENT**:
   - Start with FB condition attribute
   - Analyze image for actual wear (stains, tears, sagging, damage)
   - Read description for keywords (pet home, smoke, needs cleaning, etc.)
   - Final judgment: If image/description contradicts FB, trust visual evidence
   - RED FLAGS: holes, tears, major stains, broken parts, pet damage, smoke smell

5. **DECISION LOGIC**:
   - Calculate: resale_value = MSRP × 0.45
   - IF condition == "Poor" → decision = 0 (reject)
   - ELSE IF resale_value > $500 → decision = 1 (accept)
   - ELSE IF resale_value $450-550 → use brand + condition for judgment
   - ELSE IF budget brand (IKEA) AND resale < $400 → decision = 0 (reject)
   - ELSE → decision = 0 (reject)

6. **SKIP IF**: 
   - Both brand AND MSRP are "Unknown" (return {{"SKIP": true}})
   - OR listing price is $0-5 (marketing price) AND both brand and MSRP unknown

7. **FURNITURE TYPE** (IMAGE IS PRIMARY SOURCE OF TRUTH):
   - **LOOK AT THE IMAGE FIRST** - Title/description may be inaccurate
   - 1 = Sectional (L-shape, multiple connected pieces, modular)
   - 2 = Couch/Sofa (single long piece, 3+ seats, includes recliners)
   - 3 = Loveseat (compact, 2 seats)
   
   **ONLY SKIP IF IMAGE CLEARLY SHOWS NON-SEATING**:
   - If IMAGE shows it's clearly NOT a couch/sectional/loveseat, return {{"SKIP": true}}
   - Skip only if image shows: single chair, table, bed, ottoman by itself, pet crate
   - **DO NOT skip based on text alone** - title/description can be misleading
   - If image shows a couch/sectional/loveseat → LABEL IT (even if title is vague)
   - When in doubt about the image → LABEL IT (don't skip)

8. **UNKNOWN VALUES**: Use "UNKNOWN" (all caps) for any unknown fields

OUTPUT FORMAT (JSON only):
{{
  "sub_category_2_id": <1 or 2 or 3>,
  "brand_name": "<Standardized brand name or 'UNKNOWN'>",
  "title": "<Brand + Model/Description>",
  "msrp": <number or "UNKNOWN">,
  "condition": "<EXACTLY: Excellent|Gently Used|Fair|Poor>",
  "decision": <1 for accept, 0 for reject>,
  "confidence_brand": <0.0-1.0>,
  "confidence_msrp": <0.0-1.0>,
  "confidence_category": <0.0-1.0>,
  "msrp_source": "<description|web_exact|web_average|brand_estimate|UNKNOWN>",
  "reasoning": "<Brief explanation of decision>",
  "red_flags": ["<list of quality concerns if any>"]
}}

SKIP CONDITIONS:
1. If both brand AND MSRP are unknown with <80% confidence: {{"SKIP": true, "reason": "insufficient data"}}
2. If item is NOT seating furniture (chair, table, etc.): {{"SKIP": true, "reason": "not seating furniture"}}

Return ONLY valid JSON."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize production labeler."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.request_count = 0
        self.start_time = time.time()
        logger.info(f"Initialized ProductionFurnitureLabeler with {model_name}")
    
    def is_non_seating_listing(self, title: str, description: str) -> bool:
        """Lightweight check for OBVIOUS non-furniture (rely on image analysis for real filtering)."""
        text = (title + " " + description).lower()
        
        # Only skip VERY OBVIOUS non-furniture (be conservative, let AI decide based on image)
        obvious_non_furniture = [
            # Services/Wanted ads
            'wanted', 'looking for', 'in search of', 'wtb', 'iso',
            'delivery service', 'moving service', 'furniture assembly',
            
            # Housing (very clear indicators)
            'room for rent', 'apartment for rent', 'housing', 'sublet', 
            'roommate wanted', 'lease', 'tenant',
            
            # Pet items (very specific)
            'dog crate', 'pet kennel', 'pet carrier', 'cat carrier',
            
            # Covers ONLY (not the furniture)
            'cover only', 'just the cover', 'slipcover only',
            'selling just cover', 'cover for sale'
        ]
        
        # Only skip if very obviously not furniture
        for keyword in obvious_non_furniture:
            if keyword in text:
                return True
        
        return False  # Let AI analyze the image for everything else
    
    def standardize_brand(self, brand: str) -> str:
        """Standardize brand name."""
        if not brand or brand.lower() in ["unknown", "unkown"]:
            return "UNKNOWN"
        
        brand_lower = brand.lower().strip()
        return self.BRAND_STANDARDIZATION.get(brand_lower, brand.title())
    
    def validate_condition(self, condition: str) -> str:
        """Validate and standardize condition label - NEVER return UNKNOWN."""
        valid_conditions = ["Excellent", "Gently Used", "Fair", "Poor"]
        
        if not condition:
            return "Gently Used"  # Default if truly unknown
        
        # Normalize
        condition = condition.strip()
        
        # Check exact match
        if condition in valid_conditions:
            return condition
        
        # Try case-insensitive match
        condition_lower = condition.lower()
        for valid in valid_conditions:
            if condition_lower == valid.lower():
                return valid
        
        # Map common variations
        if "new" in condition_lower or "brand new" in condition_lower:
            return "Excellent"  # New = Excellent
        elif "excellent" in condition_lower or "like new" in condition_lower or "mint" in condition_lower:
            return "Excellent"
        elif "gently" in condition_lower or "light" in condition_lower or "good" in condition_lower:
            return "Gently Used"
        elif "fair" in condition_lower or "worn" in condition_lower or "used" in condition_lower:
            return "Fair"
        elif "poor" in condition_lower or "bad" in condition_lower or "damaged" in condition_lower:
            return "Poor"
        
        # Default to Gently Used (most common condition)
        logger.warning(f"Unmapped condition '{condition}', defaulting to 'Gently Used'")
        return "Gently Used"
    
    def rate_limit_check(self):
        """Check and enforce rate limits (4000 RPM)."""
        self.request_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed < 60:  # Within first minute
            requests_per_second = self.request_count / elapsed
            if requests_per_second > 66:  # 4000/60 = 66.67
                sleep_time = (self.request_count / 66) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        else:
            # Reset counter every minute
            self.request_count = 0
            self.start_time = time.time()
    
    def download_image(self, url: str) -> Optional[tuple]:
        """Download image."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            return (img_byte_arr.getvalue(), 'image/jpeg')
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            return None
    
    def extract_with_gemini(
        self,
        img_bytes: bytes,
        title: str,
        description: str,
        fb_condition: str,
        listing_price: float
    ) -> Optional[Dict]:
        """Extract labels using Gemini."""
        
        prompt = self.EXPERT_PROMPT_TEMPLATE.format(
            few_shot_examples=self.FEW_SHOT_EXAMPLES,
            title=title,
            description=description,
            fb_condition=fb_condition,
            listing_price=listing_price
        )
        
        try:
            self.rate_limit_check()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    prompt
                ]
            )
            
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Check for SKIP
            if result.get("SKIP"):
                logger.info(f"Skipped: {result.get('reason')}")
                return None
            
            # Validate: Skip non-seating furniture
            category = result.get('sub_category_2_id')
            if category not in [1, 2, 3]:
                logger.info(f"Skipped: Invalid category {category} (not seating)")
                return None
            
            # Double-check reasoning for non-seating mentions
            reasoning = result.get('reasoning', '').lower()
            non_seating_indicators = [
                'pet crate', 'dog crate', 'kennel', 'cat carrier',
                'headboard', 'bed frame',
                'sofa cover', 'couch cover', 'slipcover', 'cover only',
                'not a sofa', 'not a sectional', 'not a loveseat', 'not a couch',
                'not seating', 'not furniture',
                'ottoman only', 'pillow', 'cushion only',
                'table', 'desk', 'chair only', 'dining chair'
            ]
            
            for indicator in non_seating_indicators:
                if indicator in reasoning:
                    logger.info(f"Skipped: AI detected non-seating in reasoning ('{indicator}')")
                    return None
            
            # Standardize brand
            if 'brand_name' in result:
                result['brand_name'] = self.standardize_brand(result['brand_name'])
            
            # Validate condition
            if 'condition' in result:
                result['condition'] = self.validate_condition(result['condition'])
            
            # Standardize "unknown" to "UNKNOWN"
            if result.get('msrp') in ['unknown', 'Unknown', 'unkown']:
                result['msrp'] = 'UNKNOWN'
            if result.get('msrp_source') in ['unknown', 'Unknown']:
                result['msrp_source'] = 'UNKNOWN'
            
            # Validate decision logic
            msrp = result.get('msrp')
            condition = result.get('condition')
            
            if isinstance(msrp, (int, float)) and msrp != "unknown":
                resale_value = msrp * 0.45
                
                # CEO rule: Poor condition = always reject
                if condition == "Poor":
                    result['decision'] = 0
                    result['reasoning'] += " | Poor condition = auto-reject"
                # Resale > $500 = accept
                elif resale_value > 500:
                    result['decision'] = 1
                # Soft judgment around threshold
                elif 450 <= resale_value <= 550:
                    if condition in ["Excellent", "Gently Used"]:
                        result['decision'] = 1
                    else:
                        result['decision'] = 0
                else:
                    result['decision'] = 0
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON error: {e}")
            return None
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return None
    
    def web_search_msrp(self, brand: str, item_title: str, furniture_type: str = "sofa") -> Optional[float]:
        """Enhanced web search for MSRP - works with lesser-known brands like Zinus."""
        if brand == "UNKNOWN":
            return None
        
        logger.info(f"Searching web for {brand} {furniture_type} MSRP...")
        
        try:
            # Try multiple search strategies for better results
            search_queries = [
                f"{brand} {item_title} price MSRP",  # Exact model
                f"{brand} {furniture_type} retail price",  # Brand + type
                f"how much does {brand} {furniture_type} cost"  # Natural language
            ]
            
            all_prices = []
            
            for idx, query in enumerate(search_queries[:2]):  # Try first 2
                try:
                    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                    
                    response = requests.get(search_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        text = response.text
                        # Extract all price patterns
                        prices = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
                        
                        for price_str in prices[:25]:  # Check more prices
                            try:
                                clean_price = float(price_str.replace('$', '').replace(',', ''))
                                # Reasonable furniture price range
                                if 200 <= clean_price <= 15000:
                                    all_prices.append(clean_price)
                            except ValueError:
                                continue
                    
                    # Rate limit between searches
                    if idx < len(search_queries) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    logger.warning(f"Search attempt failed: {e}")
                    continue
            
            if all_prices:
                # Remove outliers and calculate median
                all_prices.sort()
                
                # If we have enough prices, remove extreme outliers
                if len(all_prices) >= 10:
                    # Remove lowest 20% and highest 20%
                    lower_cut = len(all_prices) // 5
                    upper_cut = len(all_prices) - lower_cut
                    filtered_prices = all_prices[lower_cut:upper_cut]
                else:
                    filtered_prices = all_prices
                
                if filtered_prices:
                    median = filtered_prices[len(filtered_prices) // 2]
                    logger.info(f"✓ Web search found MSRP: ${median} (from {len(all_prices)} prices)")
                    return median
            
            logger.info(f"No MSRP found via web search for {brand}")
            return None
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return None
    
    def process_item(self, row: pd.Series) -> Optional[Dict]:
        """Process single item."""
        
        photo_id = row.get('id', '')
        photo_url = row.get('primary_listing_photo.image.uri', '')
        title = str(row.get('marketplace_listing_title', ''))
        description = str(row.get('listing_details.redacted_description.text', ''))
        fb_attributes = str(row.get('listing_details.attribute_data', ''))
        listing_price = float(row.get('listing_price.amount', 0))
        
        logger.info(f"Processing: {photo_id} - {title[:50]}...")
        
        # Pre-filter: Check if this is non-seating furniture based on text
        if self.is_non_seating_listing(title, description):
            logger.info(f"Skipped: Not seating furniture (detected from title/description)")
            return None
        
        # Skip marketing listings ($0-5) unless we can determine brand or MSRP
        # We'll check this after AI extraction
        
        # Download image
        img_data = self.download_image(photo_url)
        if not img_data:
            return None
        
        img_bytes, _ = img_data
        
        # Extract with Gemini
        result = self.extract_with_gemini(
            img_bytes, title, description, fb_attributes, listing_price
        )
        
        if not result:
            return None
        
        # Check for marketing listings ($0-5) with no brand/MSRP
        brand = result.get('brand_name', 'UNKNOWN')
        msrp = result.get('msrp', 'UNKNOWN')
        
        # Try web search if we have brand but no MSRP
        if brand != 'UNKNOWN' and (msrp == 'UNKNOWN' or msrp == 'UNKOWN'):
            logger.info(f"Brand found ({brand}) but no MSRP, trying web search...")
            furniture_type = "sectional" if result.get('sub_category_2_id') == 1 else \
                           "loveseat" if result.get('sub_category_2_id') == 3 else "sofa"
            
            web_msrp = self.web_search_msrp(brand, result.get('title', ''), furniture_type)
            if web_msrp:
                result['msrp'] = web_msrp
                result['msrp_source'] = 'web_search'
                result['confidence_msrp'] = 0.70  # Lower confidence for web search
                msrp = web_msrp
        
        if listing_price <= 5:
            # Marketing price - skip if both brand and MSRP are unknown
            if brand == 'UNKNOWN' and msrp == 'UNKNOWN':
                logger.info(f"Skipped: Marketing listing (${listing_price}) with no brand/MSRP")
                return None
        
        # Format output (ensure UNKNOWN is all caps)
        output = {
            'photo_id': photo_id,
            'photo': photo_url,
            'condition': result.get('condition', 'UNKNOWN'),
            'sub_category_2_id': result.get('sub_category_2_id'),
            'title': result.get('title', 'UNKNOWN'),
            'msrp': result.get('msrp', 'UNKNOWN'),
            'brand_name': result.get('brand_name', 'UNKNOWN'),
            'confidence_brand': result.get('confidence_brand', 0.0),
            'confidence_msrp': result.get('confidence_msrp', 0.0),
            'confidence_category': result.get('confidence_category', 0.0),
            'msrp_source': result.get('msrp_source', 'UNKNOWN'),
            'reasoning': result.get('reasoning', ''),
            'red_flags': ','.join(result.get('red_flags', [])),
            'decision': result.get('decision', 0),
            # Add original data columns at the end
            'listing_title': title,
            'listing_description': description,
            'fb_attribute_condition': fb_attributes,
            'fb_listing_price': listing_price,
            'fb_is_sold': row.get('listing_details.is_sold', '')
        }
        
        logger.info(f"✓ {output['brand_name']} | MSRP: ${output['msrp']} | Decision: {output['decision']}")
        
        return output
    
    def process_batch(
        self,
        input_csv: str,
        output_csv: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        batch_size: int = 50
    ):
        """Process batch with parallel processing."""
        
        logger.info(f"\n{'='*80}")
        logger.info("PRODUCTION FURNITURE LABELING")
        logger.info(f"{'='*80}")
        
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} items")
        
        if end_idx is None:
            end_idx = len(df)
        end_idx = min(end_idx, len(df))
        
        results = []
        skipped = 0
        
        # Process with threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                future = executor.submit(self.process_item, row)
                futures.append((idx, future))
                
                # Submit in batches
                if len(futures) >= batch_size:
                    for idx, future in futures:
                        try:
                            result = future.result(timeout=30)
                            if result:
                                results.append(result)
                            else:
                                skipped += 1
                        except Exception as e:
                            logger.error(f"Error on item {idx}: {e}")
                            skipped += 1
                    
                    futures = []
                    
                    # Save progress
                    if len(results) % 100 == 0 and results:
                        pd.DataFrame(results).to_csv(output_csv, index=False)
                        logger.info(f"Progress saved: {len(results)} items")
            
            # Process remaining
            for idx, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                    else:
                        skipped += 1
                except Exception as e:
                    logger.error(f"Error on item {idx}: {e}")
                    skipped += 1
        
        # Save final results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            
            logger.info(f"\n{'='*80}")
            logger.info("PROCESSING COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Successful: {len(results)}")
            logger.info(f"Skipped: {skipped}")
            logger.info(f"Accept: {sum(1 for r in results if r['decision'] == 1)}")
            logger.info(f"Reject: {sum(1 for r in results if r['decision'] == 0)}")
            logger.info(f"\nOutput: {output_csv}")
            logger.info(f"{'='*80}\n")
        else:
            logger.error("No items successfully processed")


def generate_random_sample(input_csv: str, sample_size: int = 25, seed: Optional[int] = None) -> List[int]:
    """Generate random sample indices."""
    df = pd.read_csv(input_csv)
    total_items = len(df)
    
    # Use timestamp as seed if not provided (different each time)
    if seed is None:
        seed = int(time.time())
    
    random.seed(seed)
    sample_indices = random.sample(range(total_items), min(sample_size, total_items))
    logger.info(f"Random seed: {seed} (for reproducibility)")
    return sorted(sample_indices)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Furniture Labeling')
    parser.add_argument('--input', default='../data/clean_furniture_data.csv')
    parser.add_argument('--output', default='../data/production_labeled.csv')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--random-sample', type=int, default=0, help='Label N random samples (e.g., 25, 100)')
    parser.add_argument('--api-key', default=None)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('GOOGLE_AI_API_KEY')
    if not api_key:
        logger.error("API key required")
        sys.exit(1)
    
    labeler = ProductionFurnitureLabeler(api_key=api_key)
    
    if args.random_sample > 0:
        # Generate and process random sample
        sample_size = args.random_sample
        logger.info(f"Generating random sample of {sample_size} items...")
        sample_indices = generate_random_sample(args.input, sample_size)
        logger.info(f"Sample indices: {sample_indices}")
        
        # Process only these indices
        df = pd.read_csv(args.input)
        sample_df = df.iloc[sample_indices]
        sample_csv = '../data/random_sample_input.csv'
        sample_df.to_csv(sample_csv, index=False)
        
        labeler.process_batch(sample_csv, args.output, 0, len(sample_df))
    else:
        labeler.process_batch(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()

