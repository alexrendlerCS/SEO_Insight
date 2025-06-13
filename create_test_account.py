import os
from dotenv import load_dotenv
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_test_customer(manager_customer_id: str):
    """
    Create a test customer account under the manager account.
    
    Args:
        manager_customer_id (str): The manager account ID (without dashes)
    """
    try:
        # Initialize the client
        credentials = {
            "developer_token": os.getenv("GOOGLE_DEVELOPER_TOKEN"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
            "use_proto_plus": True
        }
        
        client = GoogleAdsClient.load_from_dict(credentials)
        customer_service = client.get_service("CustomerService")

        # Create customer object
        customer = client.get_type("Customer")
        customer.descriptive_name = "Test Account (API Only)"
        customer.currency_code = "USD"
        customer.time_zone = "America/Los_Angeles"

        # Remove dashes from manager customer ID
        manager_id = manager_customer_id.replace("-", "")
        
        logger.info(f"Creating test account under manager ID: {manager_id}")
        
        # Create the test customer
        response = customer_service.create_customer_client(
            customer_id=manager_id,
            customer_client=customer
        )
        
        logger.info(f"✅ Test account created successfully!")
        logger.info(f"Resource name: {response.resource_name}")
        
        # Extract the new customer ID from the resource name
        # Format: customers/{customer_id}
        new_customer_id = response.resource_name.split("/")[-1]
        logger.info(f"New customer ID: {new_customer_id}")
        
        # Save the new customer ID to .env file
        with open(".env", "a") as f:
            f.write(f"\nGOOGLE_TEST_CUSTOMER_ID={new_customer_id}")
        
        logger.info("✅ Added new customer ID to .env file")
        
        return new_customer_id
        
    except GoogleAdsException as ex:
        logger.error("❌ Failed to create test account:")
        for error in ex.failure.errors:
            logger.error(f"\tError with message: {error.message}")
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    logger.error(f"\t\tOn field: {field_path_element.field_name}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    # Get manager account ID from environment or use default
    manager_id = os.getenv("GOOGLE_MANAGER_ID", "194-441-4956")
    
    try:
        new_customer_id = create_test_customer(manager_id)
        print("\nNext steps:")
        print("1. Update your .env file with the new test customer ID")
        print("2. Run test_google_ads.py with the test customer ID")
    except Exception as e:
        print(f"\n❌ Failed to create test account: {str(e)}") 