"""
Local LLM integration service for generating SEO suggestions using Ollama.
"""

import requests
from typing import List, Dict, Optional
import os
import logging
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, model: str = 'llama3', api_url: str = 'http://localhost:11434/api/generate'):
        """
        Initialize Ollama client.
        
        Args:
            model (str): Ollama model to use (llama3, mistral, or codellama)
            api_url (str): Ollama API endpoint
        """
        self.model = model
        self.api_url = api_url
        self.system_prompts = {
            'keyword_suggestions': "You are an SEO expert specializing in keyword optimization and search intent analysis.",
            'meta_description': "You are an SEO expert specializing in meta description optimization.",
            'seo_suggestions': "You are an SEO expert providing detailed analysis and suggestions.",
            'cluster_analysis': "You are an SEO expert providing strategic cluster analysis."
        }
    
    def generate_ollama_response(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt (str): The input prompt
            system_prompt (str, optional): System prompt to guide the model
            temperature (float): Controls randomness (0.0 to 1.0)
            
        Returns:
            str: Generated response
        """
        try:
            # Construct the full prompt with system message if provided
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            payload = {
                'model': self.model,
                'prompt': full_prompt,
                'stream': False,
                'temperature': temperature
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            
            return response.json()['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            return ""
    
    def generate_keyword_suggestions(self, keywords: List[str], num_suggestions: int = 3) -> Dict[str, List[str]]:
        """
        Generate alternative keyword suggestions for underperforming keywords.
        
        Args:
            keywords (List[str]): List of underperforming keywords
            num_suggestions (int): Number of suggestions to generate per keyword
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping original keywords to their suggestions
        """
        try:
            prompt = f"""
            These ad keywords are underperforming:
            {', '.join(keywords)}

            For each keyword, suggest {num_suggestions} more specific, high-intent alternatives that:
            1. Target users closer to making a purchase
            2. Include specific product features or benefits
            3. Use action-oriented language
            4. Are more specific than the original keyword

            Format the response as a JSON object where:
            - Keys are the original keywords
            - Values are lists of suggested alternatives

            Example format:
            {{
                "cheap shoes": ["discount running shoes", "affordable men's sneakers", "budget footwear deals"],
                "shoe sale": ["clearance athletic shoes", "summer sneaker discounts", "limited-time shoe promotions"]
            }}
            """
            
            response = self.generate_ollama_response(
                prompt,
                system_prompt=self.system_prompts['keyword_suggestions'],
                temperature=0.7
            )
            
            # Parse the response
            try:
                # Find JSON in the response (in case model adds extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                # Return empty suggestions for failed keywords
                return {keyword: [] for keyword in keywords}
            
        except Exception as e:
            logger.error(f"Error generating keyword suggestions: {str(e)}")
            return {keyword: [] for keyword in keywords}
    
    def generate_meta_description(self, keyword: str) -> str:
        """
        Generate an SEO meta description for a given keyword.
        
        Args:
            keyword (str): The target keyword
            
        Returns:
            str: Generated meta description (max 155 characters)
        """
        try:
            prompt = f"""
            Generate a concise, 155-character SEO meta description for the keyword: "{keyword}".
            
            Requirements:
            1. Maximum 155 characters
            2. Include the keyword naturally
            3. Be actionable and relevant to searchers
            4. Highlight unique value proposition
            5. Include a clear call-to-action
            
            Return only the description text, no additional formatting or explanation.
            """
            
            description = self.generate_ollama_response(
                prompt,
                system_prompt=self.system_prompts['meta_description'],
                temperature=0.7
            )
            
            # Ensure the description is within character limit
            if len(description) > 155:
                description = description[:152] + "..."
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating meta description: {str(e)}")
            return f"Find the best {keyword} options. Compare prices, read reviews, and make an informed decision."
    
    def generate_seo_suggestions(self, keyword: str, metrics: Dict) -> Dict:
        """
        Generate SEO suggestions for a keyword based on its performance metrics.
        
        Args:
            keyword (str): The keyword to analyze
            metrics (Dict): Performance metrics for the keyword
            
        Returns:
            Dict: SEO suggestions and analysis
        """
        try:
            prompt = f"""
            Analyze the following keyword and its performance metrics to provide SEO suggestions:
            
            Keyword: {keyword}
            Metrics:
            - Impressions: {metrics.get('impressions', 'N/A')}
            - Clicks: {metrics.get('clicks', 'N/A')}
            - Cost: {metrics.get('cost', 'N/A')}
            - Conversions: {metrics.get('conversions', 'N/A')}
            - Average Position: {metrics.get('avg_position', 'N/A')}
            
            Please provide:
            1. Content suggestions
            2. On-page optimization tips
            3. Competitive analysis
            4. Long-tail keyword opportunities
            """
            
            suggestions = self.generate_ollama_response(
                prompt,
                system_prompt=self.system_prompts['seo_suggestions'],
                temperature=0.7
            )
            
            return {
                "keyword": keyword,
                "suggestions": suggestions,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating SEO suggestions: {str(e)}")
            return {
                "keyword": keyword,
                "suggestions": "Unable to generate suggestions at this time.",
                "metrics": metrics
            }
    
    def analyze_keyword_cluster(self, keywords: List[str], cluster_metrics: Dict) -> Dict:
        """
        Analyze a cluster of keywords and provide strategic recommendations.
        
        Args:
            keywords (List[str]): List of keywords in the cluster
            cluster_metrics (Dict): Aggregated metrics for the cluster
            
        Returns:
            Dict: Cluster analysis and recommendations
        """
        try:
            prompt = f"""
            Analyze the following keyword cluster and provide strategic recommendations:
            
            Keywords: {', '.join(keywords)}
            Cluster Metrics:
            - Total Impressions: {cluster_metrics.get('total_impressions', 'N/A')}
            - Average CTR: {cluster_metrics.get('avg_ctr', 'N/A')}
            - Total Cost: {cluster_metrics.get('total_cost', 'N/A')}
            - Total Conversions: {cluster_metrics.get('total_conversions', 'N/A')}
            
            Please provide:
            1. Cluster theme analysis
            2. Content strategy recommendations
            3. Budget allocation suggestions
            4. Competitive positioning advice
            """
            
            analysis = self.generate_ollama_response(
                prompt,
                system_prompt=self.system_prompts['cluster_analysis'],
                temperature=0.7
            )
            
            return {
                "keywords": keywords,
                "analysis": analysis,
                "metrics": cluster_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing keyword cluster: {str(e)}")
            return {
                "keywords": keywords,
                "analysis": "Unable to analyze cluster at this time.",
                "metrics": cluster_metrics
            } 