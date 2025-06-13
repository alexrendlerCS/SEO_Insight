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

# Constants
LANGUAGE_EN_US = "languageConstants/1000"  # English (United States)
LOCATION_US = "geoTargetConstants/2840"    # United States

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
        language_id: str = LANGUAGE_EN_US,
        location_ids: List[str] = [LOCATION_US],
        customer_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get keyword ideas and metrics from Google Ads API.
        
        Args:
            keywords: List of seed keywords to get ideas for
            language_id: Language resource name (default: English US)
            location_ids: List of location resource names (default: United States)
            customer_id: Google Ads customer ID (optional)
            
        Returns:
            DataFrame containing keyword ideas and metrics
            
        Raises:
            ValueError: If validation fails for resource names or keywords
            
        Note:
            CPC (Cost Per Click) data is not available in the Google Ads API v20
            GenerateKeywordIdeasRequest. Only search volume and competition metrics
            are provided.
        """
        try:
            # Validate keywords
            if not keywords:
                raise ValueError("Keywords list cannot be empty")
            
            # Validate resource name formats
            def validate_resource_name(resource_name: str, resource_type: str) -> None:
                if not resource_name.startswith(f"{resource_type}/"):
                    raise ValueError(
                        f"Invalid {resource_type} format. Expected '{resource_type}/{{id}}', "
                        f"got '{resource_name}'"
                    )
            
            # Validate language resource name
            validate_resource_name(language_id, "languageConstants")
            
            # Validate location resource names
            for location_id in location_ids:
                validate_resource_name(location_id, "geoTargetConstants")
            
            # Log request parameters
            logger.info("🔍 Request Parameters:")
            logger.info(f"  Language: {language_id}")
            logger.info(f"  Keywords: {', '.join(keywords)}")
            logger.info(f"  Customer ID: {customer_id or 'Not provided'}")
            logger.info(f"  Location IDs: {', '.join(location_ids)}")

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
                # Safely extract competition level
                try:
                    competition = (
                        idea.keyword_idea_metrics.competition.name 
                        if idea.keyword_idea_metrics.competition 
                        else "UNSPECIFIED"
                    )
                    logger.debug(f"Competition level for '{idea.text}': {competition}")
                except Exception as e:
                    logger.warning(f"Failed to extract competition level for '{idea.text}': {str(e)}")
                    competition = "UNSPECIFIED"

                results.append({
                    'keyword': idea.text,
                    'search_volume': idea.keyword_idea_metrics.avg_monthly_searches,
                    'competition': competition,
                    'competition_index': idea.keyword_idea_metrics.competition_index
                })

            logger.info(f"✅ Successfully retrieved {len(results)} keyword ideas")
            return pd.DataFrame(results)

        except ValueError as ve:
            logger.error(f"❌ Validation error: {str(ve)}")
            raise
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
            
        Raises:
            ValueError: If keywords list is empty
            
        Note:
            CPC (Cost Per Click) data is not available in the Google Ads API v20
            GenerateKeywordHistoricalMetricsRequest. Only search volume, competition,
            and bid ranges are provided.
        """
        try:
            # Validate keywords
            if not keywords:
                raise ValueError("Keywords list cannot be empty")
            
            # Log request parameters
            logger.info("🔍 Request Parameters:")
            logger.info(f"  Keywords: {', '.join(keywords)}")
            logger.info(f"  Customer ID: {customer_id or 'Not provided'}")

            keyword_plan_idea_service = self._client.get_service("KeywordPlanIdeaService")
            
            # Create request
            request = self._client.get_type("GenerateKeywordHistoricalMetricsRequest")
            request.customer_id = customer_id
            request.keywords.extend(keywords)  # Directly extend with strings

            # Get metrics
            response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)

            # Process results
            results = []
            for result in response.results:
                metrics = result.keyword_metrics
                
                # Safely extract competition level
                try:
                    competition = (
                        metrics.competition.name 
                        if metrics.competition 
                        else "UNSPECIFIED"
                    )
                    logger.debug(f"Competition level for '{result.text}': {competition}")
                except Exception as e:
                    logger.warning(f"Failed to extract competition level for '{result.text}': {str(e)}")
                    competition = "UNSPECIFIED"

                # Safely extract bid ranges
                try:
                    low_bid = (
                        metrics.low_top_of_page_bid_micros / 1_000_000 
                        if metrics.low_top_of_page_bid_micros 
                        else 0.0
                    )
                    high_bid = (
                        metrics.high_top_of_page_bid_micros / 1_000_000 
                        if metrics.high_top_of_page_bid_micros 
                        else 0.0
                    )
                    logger.debug(f"Bid range for '{result.text}': ${low_bid:.2f} - ${high_bid:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to extract bid range for '{result.text}': {str(e)}")
                    low_bid = high_bid = 0.0

                # Safely extract search volume
                try:
                    search_volume = metrics.avg_monthly_searches or 0
                    logger.debug(f"Search volume for '{result.text}': {search_volume:,}")
                except Exception as e:
                    logger.warning(f"Failed to extract search volume for '{result.text}': {str(e)}")
                    search_volume = 0

                # Safely extract competition index
                try:
                    competition_index = metrics.competition_index or 0
                    logger.debug(f"Competition index for '{result.text}': {competition_index}")
                except Exception as e:
                    logger.warning(f"Failed to extract competition index for '{result.text}': {str(e)}")
                    competition_index = 0

                results.append({
                    'keyword': result.text,
                    'avg_monthly_searches': search_volume,
                    'competition': competition,
                    'competition_index': competition_index,
                    'low_top_of_page_bid': low_bid,
                    'high_top_of_page_bid': high_bid
                })

            logger.info(f"✅ Successfully retrieved metrics for {len(results)} keywords")
            return pd.DataFrame(results)

        except ValueError as ve:
            logger.error(f"❌ Validation error: {str(ve)}")
            raise
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