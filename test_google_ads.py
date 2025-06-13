import os
import logging
from dotenv import load_dotenv
from services.google_ads_service import GoogleAdsService
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

EXPORT_DIR = "exports"

def ensure_export_dir():
    """Ensure the export directory exists."""
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        logger.info(f"📁 Created export directory: {EXPORT_DIR}")

def file_exists_and_log(filepath):
    """Check if a file exists and log its contents."""
    if os.path.isfile(filepath):
        logger.info(f"📄 CSV exported: {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"✅ Preview of exported CSV ({filepath}):")
            print(df.head())
        except Exception as e:
            logger.warning(f"⚠️ Could not preview CSV: {str(e)}")
    else:
        logger.warning(f"❌ CSV file not found at: {filepath}")

def test_google_ads_integration():
    """Test the Google Ads API integration, including export functionality."""
    try:
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

        logger.info("Initializing Google Ads Service...")
        ads_service = GoogleAdsService()

        customer_id = os.getenv("GOOGLE_TEST_CUSTOMER_ID")
        test_keywords = ["digital marketing", "seo services", "content marketing"]
        ensure_export_dir()

        # Keyword Ideas Export
        ideas_export_path = os.path.join(EXPORT_DIR, "keyword_ideas.csv")
        logger.info(f"Getting keyword ideas for: {test_keywords}")
        keyword_ideas = ads_service.get_keyword_ideas(
            test_keywords,
            customer_id=customer_id,
            export_path=ideas_export_path
        )
        logger.info(f"✅ Retrieved {len(keyword_ideas)} keyword ideas")
        file_exists_and_log(ideas_export_path)

        # Keyword Metrics Export
        metrics_export_path = os.path.join(EXPORT_DIR, "keyword_metrics.csv")
        logger.info("Getting keyword metrics...")
        keyword_metrics = ads_service.get_keyword_metrics(
            test_keywords,
            customer_id=customer_id,
            export_path=metrics_export_path
        )
        logger.info(f"✅ Retrieved metrics for {len(keyword_metrics)} keywords")
        file_exists_and_log(metrics_export_path)

        return True

    except ValueError as ve:
        logger.error(f"❌ Configuration error: {str(ve)}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_google_ads_integration()
