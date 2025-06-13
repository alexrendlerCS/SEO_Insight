import os
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleAdsService:
    def __init__(self):
        load_dotenv()
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google Ads API client with credentials."""
        try:
            credentials = {
                "developer_token": os.getenv("GOOGLE_DEVELOPER_TOKEN"),
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
                "use_proto_plus": True
            }
            
            self._client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("✅ Google Ads API client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Ads API client: {str(e)}")
            raise

    def get_keyword_ideas(
        self,
        keywords: List[str],
        language_id: str = "1000",  # English
        location_ids: List[str] = ["2840"],  # United States
        customer_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get keyword ideas and metrics from Google Ads API.
        
        Args:
            keywords: List of seed keywords to get ideas for
            language_id: Language ID (default: English)
            location_ids: List of location IDs (default: United States)
            customer_id: Google Ads customer ID (optional)
            
        Returns:
            DataFrame containing keyword ideas and metrics
        """
        try:
            keyword_plan_idea_service = self._client.get_service("KeywordPlanIdeaService")
            keyword_competition_level_enum = self._client.enums.KeywordPlanCompetitionLevelEnum

            # Create keyword seed
            keyword_seeds = []
            for keyword in keywords:
                seed = self._client.get_type("KeywordSeed")
                seed.keywords.append(keyword)
                keyword_seeds.append(seed)

            # Create request
            request = self._client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = customer_id
            request.language = language_id
            request.geo_target_constants = location_ids
            request.keyword_seed.keywords.extend(keywords)
            request.keyword_plan_network = self._client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS

            # Get keyword ideas
            keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)

            # Process results
            results = []
            for idea in keyword_ideas:
                results.append({
                    'keyword': idea.text,
                    'search_volume': idea.keyword_idea_metrics.avg_monthly_searches,
                    'competition': keyword_competition_level_enum.KeywordPlanCompetitionLevel.Name(
                        idea.keyword_idea_metrics.competition
                    ),
                    'cpc': idea.keyword_idea_metrics.average_cpc.micros / 1_000_000,
                    'competition_index': idea.keyword_idea_metrics.competition_index
                })

            return pd.DataFrame(results)

        except GoogleAdsException as ex:
            logger.error(f"❌ Google Ads API error: {ex}")
            for error in ex.failure.errors:
                logger.error(f"\tError with message: {error.message}")
                if error.location:
                    for field_path_element in error.location.field_path_elements:
                        logger.error(f"\t\tOn field: {field_path_element.field_name}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise

    def get_keyword_metrics(
        self,
        keywords: List[str],
        customer_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical metrics for specific keywords.
        
        Args:
            keywords: List of keywords to get metrics for
            customer_id: Google Ads customer ID (optional)
            
        Returns:
            DataFrame containing keyword metrics
        """
        try:
            keyword_plan_idea_service = self._client.get_service("KeywordPlanIdeaService")
            
            # Create request
            request = self._client.get_type("GenerateKeywordHistoricalMetricsRequest")
            request.customer_id = customer_id
            
            # Add keywords to request
            for keyword in keywords:
                keyword_idea = self._client.get_type("KeywordPlanKeyword")
                keyword_idea.text = keyword
                request.keywords.append(keyword_idea)

            # Get metrics
            response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)

            # Process results
            results = []
            for result in response.results:
                metrics = result.keyword_idea_metrics
                results.append({
                    'keyword': result.text,
                    'avg_monthly_searches': metrics.avg_monthly_searches,
                    'competition': metrics.competition.name,
                    'cpc': metrics.average_cpc.micros / 1_000_000,
                    'competition_index': metrics.competition_index,
                    'low_top_of_page_bid': metrics.low_top_of_page_bid.micros / 1_000_000,
                    'high_top_of_page_bid': metrics.high_top_of_page_bid.micros / 1_000_000
                })

            return pd.DataFrame(results)

        except GoogleAdsException as ex:
            logger.error(f"❌ Google Ads API error: {ex}")
            for error in ex.failure.errors:
                logger.error(f"\tError with message: {error.message}")
                if error.location:
                    for field_path_element in error.location.field_path_elements:
                        logger.error(f"\t\tOn field: {field_path_element.field_name}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise 