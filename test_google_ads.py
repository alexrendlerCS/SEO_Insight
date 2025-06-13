import os
from dotenv import load_dotenv
from services.google_ads_service import GoogleAdsService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_google_ads_integration():
    """Test the Google Ads API integration."""
    try:
        # Verify required environment variables
        required_vars = [
            "GOOGLE_DEVELOPER_TOKEN",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
            "GOOGLE_REFRESH_TOKEN",
            "GOOGLE_TEST_CUSTOMER_ID"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize the service
        logger.info("Initializing Google Ads Service...")
        ads_service = GoogleAdsService()
        
        # Get test customer ID from environment
        customer_id = os.getenv("GOOGLE_TEST_CUSTOMER_ID")
        logger.info(f"Using test customer ID: {customer_id}")
        
        # Test keyword ideas
        test_keywords = ["digital marketing", "seo services", "content marketing"]
        logger.info(f"Getting keyword ideas for: {test_keywords}")
        
        keyword_ideas = ads_service.get_keyword_ideas(
            test_keywords,
            customer_id=customer_id
        )
        logger.info(f"✅ Successfully retrieved {len(keyword_ideas)} keyword ideas")
        logger.info("\nSample keyword ideas:")
        print(keyword_ideas.head())
        
        # Test keyword metrics
        logger.info("\nGetting metrics for test keywords...")
        keyword_metrics = ads_service.get_keyword_metrics(
            test_keywords,
            customer_id=customer_id
        )
        logger.info(f"✅ Successfully retrieved metrics for {len(keyword_metrics)} keywords")
        logger.info("\nKeyword metrics:")
        print(keyword_metrics)
        
        return True
        
    except ValueError as ve:
        logger.error(f"❌ Configuration error: {str(ve)}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_google_ads_integration() 