"""
HIGH-PERFORMANCE Parallel Furniture Labeler
--------------------------------------------
Maximizes throughput for gemini-2.5-flash-lite Tier 1 limits:
- 4000 RPM (66 requests/second)
- 4M TPM (tokens per minute)
- Unlimited RPD

Processes 37,047 items in ~15-20 minutes (vs 2-3 hours sequential)
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
from typing import Dict, Optional, List
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'parallel_labeling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, rpm_limit: int = 3500, tpm_limit: int = 3500000):
        """
        Initialize rate limiter.
        
        Args:
            rpm_limit: Requests per minute (set to 3500 to stay under 4000)
            tpm_limit: Tokens per minute (set to 3.5M to stay under 4M)
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times = deque()
        self.token_counts = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 2000):
        """Wait if we're approaching rate limits."""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Remove old requests (older than 1 minute)
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            while self.token_counts and self.token_counts[0][0] < minute_ago:
                self.token_counts.popleft()
            
            # Check RPM
            current_rpm = len(self.request_times)
            if current_rpm >= self.rpm_limit:
                sleep_time = self.request_times[0] - minute_ago
                if sleep_time > 0:
                    logger.debug(f"RPM limit reached, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    return self.wait_if_needed(estimated_tokens)
            
            # Check TPM
            current_tpm = sum(count for _, count in self.token_counts)
            if current_tpm + estimated_tokens >= self.tpm_limit:
                sleep_time = 2
                logger.debug(f"TPM approaching limit, sleeping {sleep_time}s")
                time.sleep(sleep_time)
                return self.wait_if_needed(estimated_tokens)
            
            # Record this request
            self.request_times.append(now)
            self.token_counts.append((now, estimated_tokens))


class ParallelFurnitureLabeler:
    """High-performance parallel furniture labeler."""
    
    # Brand standardization
    BRAND_STANDARDIZATION = {
        'crate & barrel': 'Crate & Barrel', 'crate and barrel': 'Crate & Barrel',
        'cb': 'Crate & Barrel', 'cb2': 'CB2',
        'west elm': 'West Elm', 'westelm': 'West Elm',
        'pottery barn': 'Pottery Barn', 'potterybarn': 'Pottery Barn',
        'room & board': 'Room & Board', 'room and board': 'Room & Board',
        'restoration hardware': 'Restoration Hardware', 'rh': 'Restoration Hardware',
        'ashley': 'Ashley Furniture', 'ashley furniture': 'Ashley Furniture',
        'la-z-boy': 'La-Z-Boy', 'lazyboy': 'La-Z-Boy',
        'ikea': 'IKEA', 'wayfair': 'Wayfair', 'article': 'Article',
        'joybird': 'Joybird', 'burrow': 'Burrow',
    }
    
    FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Title: "Room & Board Sectional"
→ {{"sub_category_2_id": 1, "brand_name": "Room & Board", "msrp": 3200, "condition": "Excellent", "decision": 1}}

EXAMPLE 2:
Title: "Zinus Loveseat"
→ {{"sub_category_2_id": 3, "brand_name": "Zinus", "msrp": 450, "condition": "Gently Used", "decision": 0}}

EXAMPLE 3:
Title: "Ashley Furniture Sofa - paid $1200"
→ {{"sub_category_2_id": 2, "brand_name": "Ashley Furniture", "msrp": 1200, "condition": "Gently Used", "decision": 1}}
"""
    
    CONCISE_PROMPT = """Analyze furniture listing. Extract brand (even if lesser-known like Zinus), MSRP, condition, type.

Title: {title}
Description: {description}
FB Condition: {fb_condition}
Listing Price: ${listing_price}

Rules:
- Extract brand from title (e.g., "Zinus Loveseat" → brand="Zinus")
- MSRP from description or estimate based on brand
- Condition: ONLY "Excellent", "Gently Used", "Fair", or "Poor" (if FB says "New"→"Excellent")
- Category: 1=Sectional(L-shape), 2=Couch(single piece), 3=Loveseat(2-seat)
- Decision: 1 if MSRP>$1000 OR good brand, 0 if Poor condition OR MSRP<$1000
- SKIP if image shows: single chair, table, pet crate, bed, not seating furniture

JSON only:
{{"sub_category_2_id":<1/2/3>, "brand_name":"<name or UNKNOWN>", "title":"<full title>", "msrp":<number or "UNKNOWN">, "condition":"<Excellent|Gently Used|Fair|Poor>", "decision":<0/1>, "confidence_brand":<0-1>, "confidence_msrp":<0-1>, "msrp_source":"<description|web_search|UNKNOWN>", "reasoning":"<brief>", "red_flags":[]}}

If not seating: {{"SKIP":true}}
If both brand+MSRP unknown: {{"SKIP":true}}"""
    
    def __init__(self, api_key: str, max_workers: int = 100):
        """
        Initialize parallel labeler.
        
        Args:
            api_key: Google AI API key
            max_workers: Max concurrent threads (100 for high throughput)
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(rpm_limit=3500, tpm_limit=3500000)
        self.max_workers = max_workers
        logger.info(f"Initialized ParallelFurnitureLabeler with {max_workers} workers")
    
    def download_image(self, url: str) -> Optional[tuple]:
        """Download image quickly."""
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=80)
            return (img_byte_arr.getvalue(), 'image/jpeg')
        except Exception:
            return None
    
    def standardize_brand(self, brand: str) -> str:
        """Standardize brand name."""
        if not brand or brand.lower() in ["unknown", "unkown"]:
            return "UNKNOWN"
        brand_lower = brand.lower().strip()
        return self.BRAND_STANDARDIZATION.get(brand_lower, brand.title())
    
    def validate_condition(self, condition: str) -> str:
        """Ensure valid condition label."""
        valid_conditions = ["Excellent", "Gently Used", "Fair", "Poor"]
        
        if not condition:
            return "Gently Used"
        
        condition = condition.strip()
        if condition in valid_conditions:
            return condition
        
        condition_lower = condition.lower()
        if "new" in condition_lower or "excellent" in condition_lower or "mint" in condition_lower:
            return "Excellent"
        elif "gently" in condition_lower or "light" in condition_lower or "good" in condition_lower:
            return "Gently Used"
        elif "fair" in condition_lower or "worn" in condition_lower or "used" in condition_lower:
            return "Fair"
        elif "poor" in condition_lower or "bad" in condition_lower or "damaged" in condition_lower:
            return "Poor"
        
        return "Gently Used"
    
    def extract_with_gemini(self, img_bytes: bytes, title: str, description: str, 
                           fb_condition: str, listing_price: float) -> Optional[Dict]:
        """Extract labels with Gemini."""
        
        # Wait for rate limit
        self.rate_limiter.wait_if_needed(estimated_tokens=2000)
        
        prompt = self.CONCISE_PROMPT.format(
            title=title[:200],  # Truncate for efficiency
            description=description[:500],
            fb_condition=fb_condition,
            listing_price=listing_price
        )
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-lite',
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
                return None
            
            # Validate category
            if result.get('sub_category_2_id') not in [1, 2, 3]:
                return None
            
            # Check reasoning for non-seating
            reasoning = result.get('reasoning', '').lower()
            non_seating_indicators = [
                'pet crate', 'dog crate', 'kennel', 'cat carrier',
                'headboard', 'bed frame', 'mattress',
                'sofa cover', 'couch cover', 'not a sofa', 'not a couch',
                'not seating', 'table', 'chair only', 'ottoman only'
            ]
            
            for indicator in non_seating_indicators:
                if indicator in reasoning:
                    return None
            
            # Standardize
            result['brand_name'] = self.standardize_brand(result.get('brand_name', 'UNKNOWN'))
            result['condition'] = self.validate_condition(result.get('condition', 'Gently Used'))
            
            # Fix UNKNOWN capitalization
            if result.get('msrp') in ['unknown', 'Unknown']:
                result['msrp'] = 'UNKNOWN'
            
            return result
            
        except Exception as e:
            logger.debug(f"Extraction error: {e}")
            return None
    
    def process_item(self, row: pd.Series) -> Optional[Dict]:
        """Process single item."""
        
        photo_id = row.get('id', '')
        photo_url = row.get('primary_listing_photo.image.uri', '')
        title = str(row.get('marketplace_listing_title', ''))
        description = str(row.get('listing_details.redacted_description.text', ''))
        fb_attributes = str(row.get('listing_details.attribute_data', ''))
        listing_price = float(row.get('listing_price.amount', 0))
        is_sold = row.get('listing_details.is_sold', '')
        
        # Download image
        img_data = self.download_image(photo_url)
        if not img_data:
            return None
        
        img_bytes, _ = img_data
        
        # Extract with Gemini
        result = self.extract_with_gemini(img_bytes, title, description, fb_attributes, listing_price)
        
        if not result:
            return None
        
        # Check marketing listings
        brand = result.get('brand_name', 'UNKNOWN')
        msrp = result.get('msrp', 'UNKNOWN')
        
        if listing_price <= 5 and brand == 'UNKNOWN' and msrp == 'UNKNOWN':
            return None
        
        # Format output
        return {
            'photo_id': photo_id,
            'photo': photo_url,
            'condition': result.get('condition', 'Gently Used'),
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
            'listing_title': title,
            'listing_description': description,
            'fb_attribute_condition': fb_attributes,
            'fb_listing_price': listing_price,
            'fb_is_sold': is_sold
        }
    
    def process_batch_parallel(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Dict]:
        """Process batch with maximum parallelization."""
        
        results = []
        skipped = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx in range(start_idx, end_idx):
                if idx < len(df):
                    row = df.iloc[idx]
                    future = executor.submit(self.process_item, row)
                    future_to_idx[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                    else:
                        skipped += 1
                except Exception as e:
                    errors += 1
                    logger.debug(f"Error on item {idx}: {e}")
                
                # Progress update every 50 items
                total_processed = len(results) + skipped + errors
                if total_processed % 50 == 0:
                    logger.info(f"Progress: {total_processed}/{end_idx-start_idx} | Success: {len(results)} | Skip: {skipped} | Error: {errors}")
        
        return results
    
    def process_full_dataset(self, input_csv: str, output_csv: str, 
                            batch_size: int = 500, resume: bool = True):
        """
        Process full dataset with parallel processing and progress saving.
        
        Args:
            input_csv: Input CSV path
            output_csv: Output CSV path
            batch_size: Items per batch for progress saving
            resume: Resume from existing output if available
        """
        logger.info(f"\n{'='*80}")
        logger.info("HIGH-PERFORMANCE PARALLEL FURNITURE LABELING")
        logger.info(f"{'='*80}")
        logger.info(f"Input: {input_csv}")
        logger.info(f"Output: {output_csv}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"Target RPM: 3500 (limit: 4000)")
        logger.info(f"{'='*80}\n")
        
        # Load data
        df = pd.read_csv(input_csv)
        total_items = len(df)
        logger.info(f"Total items to process: {total_items}")
        
        # Check for existing output (resume capability)
        start_from = 0
        all_results = []
        
        if resume and os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                all_results = existing_df.to_dict('records')
                start_from = len(all_results)
                logger.info(f"Resuming from item {start_from} (found {len(all_results)} existing labels)")
            except Exception as e:
                logger.warning(f"Could not resume: {e}")
        
        # Process in batches
        for batch_start in range(start_from, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"BATCH: Items {batch_start} to {batch_end} ({batch_end - batch_start} items)")
            logger.info(f"{'='*80}")
            
            batch_start_time = time.time()
            
            # Process batch in parallel
            batch_results = self.process_batch_parallel(df, batch_start, batch_end)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            items_per_min = (batch_end - batch_start) / (batch_time / 60)
            
            logger.info(f"\nBatch complete in {batch_time:.1f}s ({items_per_min:.1f} items/min)")
            logger.info(f"Total labeled so far: {len(all_results)}")
            
            # Save progress
            if all_results:
                results_df = pd.DataFrame(all_results)
                # Clean unicode surrogates before saving
                for col in results_df.select_dtypes(include=['object']).columns:
                    results_df[col] = results_df[col].astype(str).apply(
                        lambda x: x.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                    )
                results_df.to_csv(output_csv, index=False, encoding='utf-8')
                logger.info(f"Progress saved to {output_csv}")
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("PROCESSING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Total items processed: {total_items}")
        logger.info(f"Successfully labeled: {len(all_results)}")
        logger.info(f"Success rate: {len(all_results)/total_items*100:.1f}%")
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            accept_count = sum(1 for r in all_results if r.get('decision') == 1)
            logger.info(f"Accept: {accept_count} ({accept_count/len(all_results)*100:.1f}%)")
            logger.info(f"Reject: {len(all_results) - accept_count}")
            
            # Brand coverage
            has_brand = sum(1 for r in all_results if r.get('brand_name') != 'UNKNOWN')
            logger.info(f"Brand found: {has_brand} ({has_brand/len(all_results)*100:.1f}%)")
            
            # MSRP coverage
            has_msrp = sum(1 for r in all_results if r.get('msrp') != 'UNKNOWN')
            logger.info(f"MSRP found: {has_msrp} ({has_msrp/len(all_results)*100:.1f}%)")
        
        logger.info(f"\nOutput: {output_csv}")
        logger.info(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel Furniture Labeling')
    parser.add_argument('--input', default='../data/clean_furniture_data.csv')
    parser.add_argument('--output', default='../data/FULL_DATASET_LABELED.csv')
    parser.add_argument('--workers', type=int, default=100, help='Max concurrent workers')
    parser.add_argument('--no-resume', action='store_true', help='Start from scratch')
    parser.add_argument('--api-key', default=None)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('GOOGLE_AI_API_KEY')
    if not api_key:
        logger.error("API key required")
        sys.exit(1)
    
    labeler = ParallelFurnitureLabeler(api_key=api_key, max_workers=args.workers)
    labeler.process_full_dataset(
        input_csv=args.input,
        output_csv=args.output,
        batch_size=500,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()




