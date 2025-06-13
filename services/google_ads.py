"""
Google Ads API integration service.
Handles authentication and data fetching from Google Ads API.
"""

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

class GoogleAdsService:
    def __init__(self, use_mock_data: bool = False):
        """
        Initialize Google Ads client.
        
        Args:
            use_mock_data (bool): Whether to use mock data instead of real API calls
        """
        self.use_mock_data = use_mock_data
        load_dotenv()
        
        if not use_mock_data:
            self.client = self._initialize_client()
    
    def _initialize_client(self) -> Optional[GoogleAdsClient]:
        """
        Initialize Google Ads client using OAuth credentials.
        
        Returns:
            Optional[GoogleAdsClient]: Initialized client or None if authentication fails
        """
        try:
            credentials = {
                "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
                "use_proto_plus": True
            }
            
            return GoogleAdsClient.load_from_dict(credentials)
        except Exception as e:
            print(f"Failed to initialize Google Ads client: {str(e)}")
            return None
    
    def get_keyword_metrics(self, account_id: str, date_range: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Fetch keyword performance data from Google Ads.
        
        Args:
            account_id (str): Google Ads account ID
            date_range (Optional[Dict[str, str]]): Date range for the report
            
        Returns:
            pd.DataFrame: Keyword performance data
        """
        if self.use_mock_data:
            return self._get_mock_data()
        
        if not self.client:
            print("Google Ads client not initialized. Falling back to mock data.")
            return self._get_mock_data()
        
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            # Set default date range if not provided
            if not date_range:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                date_range = {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                }
            
            query = """
                SELECT
                    keyword_view.keyword.text,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.average_position,
                    metrics.ctr
                FROM keyword_view
                WHERE segments.date BETWEEN '{}' AND '{}'
            """.format(date_range['start'], date_range['end'])
            
            response = ga_service.search(customer_id=account_id, query=query)
            
            # Process and return data
            data = []
            for row in response:
                data.append({
                    'keyword': row.keyword_view.keyword.text,
                    'impressions': row.metrics.impressions,
                    'clicks': row.metrics.clicks,
                    'cost': row.metrics.cost_micros / 1_000_000,
                    'conversions': row.metrics.conversions,
                    'avg_position': row.metrics.average_position,
                    'CTR': row.metrics.ctr
                })
            
            return pd.DataFrame(data)
            
        except GoogleAdsException as ex:
            print(f'Request with ID "{ex.request_id}" failed with status '
                  f'"{ex.error.code().name}" and includes the following errors:')
            for error in ex.failure.errors:
                print(f'\tError with message "{error.message}".')
                if error.location:
                    for field_path_element in error.location.field_path_elements:
                        print(f'\t\tOn field: {field_path_element.field_name}')
            print("Falling back to mock data.")
            return self._get_mock_data()
    
    def _get_mock_data(self) -> pd.DataFrame:
        """
        Get mock keyword performance data.
        
        Returns:
            pd.DataFrame: Mock keyword performance data
        """
        try:
            # Try to load from data directory first
            mock_data_path = os.path.join('data', 'mock_keyword_data.csv')
            if not os.path.exists(mock_data_path):
                # Fallback to root directory
                mock_data_path = 'mock_keyword_data.csv'
            
            return pd.read_csv(mock_data_path)
        except Exception as e:
            print(f"Error loading mock data: {str(e)}")
            # Return minimal mock data if file loading fails
            return pd.DataFrame({
                'keyword': ['sample keyword 1', 'sample keyword 2'],
                'clicks': [10, 20],
                'impressions': [100, 200],
                'CTR': [0.1, 0.1],
                'conversions': [1, 2],
                'cost': [50.0, 100.0],
                'avg_position': [2.0, 3.0]
            }) 