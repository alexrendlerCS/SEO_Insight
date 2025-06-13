"""
Case study test script for SEO keyword analysis and suggestions.
Runs a full pipeline to analyze low-performing keywords and generate suggestions.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CaseStudyGenerator:
    def __init__(self, ollama_url: str = 'http://localhost:11434/api/generate'):
        """Initialize the case study generator."""
        self.ollama_url = ollama_url
        self.ollama_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and available."""
        try:
            response = requests.get(self.ollama_url.replace('/generate', '/tags'))
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama is not running. Will use mock suggestions.")
            return False
    
    def generate_ollama_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from Ollama model."""
        if not self.ollama_available:
            return self._generate_mock_response(prompt)
            
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {
                'model': 'llama3',
                'prompt': full_prompt,
                'stream': False
            }
            
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response when Ollama is not available."""
        if "suggest" in prompt.lower():
            return json.dumps({
                "suggestions": [
                    f"improved_{prompt.split()[0]}_1",
                    f"better_{prompt.split()[0]}_2",
                    f"optimized_{prompt.split()[0]}_3"
                ]
            })
        else:
            return f"Meta description for {prompt.split()[0]} with compelling call-to-action."
    
    def generate_keyword_suggestions(self, keyword: str) -> List[str]:
        """Generate alternative keywords using LLM."""
        prompt = f"""Generate 2-3 high-intent alternative keywords for: {keyword}
        Focus on commercial intent and user search behavior.
        Return as JSON array of strings."""
        
        system_prompt = "You are an SEO expert specializing in keyword optimization."
        
        try:
            response = self.generate_ollama_response(prompt, system_prompt)
            suggestions = json.loads(response)
            return suggestions.get('suggestions', [])[:3]
        except Exception as e:
            logger.error(f"Error generating suggestions for {keyword}: {str(e)}")
            return [f"improved_{keyword}", f"better_{keyword}"]
    
    def generate_meta_description(self, keyword: str) -> str:
        """Generate meta description using LLM."""
        prompt = f"""Generate a compelling 155-character meta description for: {keyword}
        Include a clear value proposition and call-to-action."""
        
        system_prompt = "You are an SEO expert specializing in meta description optimization."
        
        try:
            response = self.generate_ollama_response(prompt, system_prompt)
            return response[:155]  # Ensure we don't exceed character limit
        except Exception as e:
            logger.error(f"Error generating meta description for {keyword}: {str(e)}")
            return f"Find the best {keyword} solutions. Expert advice and top-rated options available now."
    
    def simulate_ctr_uplift(self, original_ctr: float) -> Tuple[float, float]:
        """Simulate CTR uplift with randomization."""
        base_uplift = np.random.uniform(0.8, 2.0)  # Base uplift between 0.8% and 2.0%
        random_factor = np.random.uniform(0.9, 1.1)  # Random factor between 0.9 and 1.1
        uplift = base_uplift * random_factor
        new_ctr = original_ctr + uplift
        return new_ctr, uplift

def load_mock_data() -> pd.DataFrame:
    """Load mock keyword data from CSV."""
    try:
        mock_data_path = os.path.join('data', 'mock_keyword_data.csv')
        if not os.path.exists(mock_data_path):
            mock_data_path = 'mock_keyword_data.csv'
        df = pd.read_csv(mock_data_path)
        logger.info(f"Successfully loaded mock data with {len(df)} keywords")
        return df
    except Exception as e:
        logger.error(f"Error loading mock data: {str(e)}")
        raise

def filter_low_performing_keywords(df: pd.DataFrame, ctr_threshold: float = 1.0) -> pd.DataFrame:
    """Filter keywords with CTR below threshold."""
    low_perf_df = df[df['CTR'] < ctr_threshold].copy()
    logger.info(f"Found {len(low_perf_df)} keywords with CTR < {ctr_threshold}%")
    return low_perf_df

def run_case_study() -> None:
    """Run the full case study pipeline."""
    try:
        # Initialize generator
        generator = CaseStudyGenerator()
        
        # Load and filter data
        df = load_mock_data()
        low_perf_df = filter_low_performing_keywords(df)
        
        # Prepare results
        results = []
        
        # Process each low-performing keyword
        for _, row in low_perf_df.iterrows():
            keyword = row['keyword']
            original_ctr = row['CTR']
            
            # Generate suggestions and meta description
            suggested_keywords = generator.generate_keyword_suggestions(keyword)
            meta_description = generator.generate_meta_description(keyword)
            
            # Simulate CTR uplift
            estimated_ctr, ctr_uplift = generator.simulate_ctr_uplift(original_ctr)
            
            # Add to results
            results.append({
                'keyword': keyword,
                'original_ctr': original_ctr,
                'suggested_keywords': ' | '.join(suggested_keywords),
                'meta_description': meta_description,
                'estimated_ctr': estimated_ctr,
                'ctr_uplift': ctr_uplift
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = 'case_study_results.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"Case study results saved to {output_path}")
        
        # Print summary
        print("\nCase Study Summary:")
        print(f"Total keywords analyzed: {len(results_df)}")
        print(f"Average CTR uplift: {results_df['ctr_uplift'].mean():.2f}%")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error running case study: {str(e)}")
        raise

if __name__ == "__main__":
    run_case_study() 