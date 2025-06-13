"""
Google OAuth authentication utilities.
"""

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import os
import json
from typing import Optional

class GoogleAuth:
    def __init__(self, credentials_path: str, token_path: str):
        """
        Initialize Google authentication handler.
        
        Args:
            credentials_path (str): Path to the OAuth 2.0 client credentials file
            token_path (str): Path to store/load the token
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.SCOPES = [
            'https://www.googleapis.com/auth/adwords',
            'https://www.googleapis.com/auth/analytics.readonly'
        ]
    
    def get_credentials(self) -> Optional[Credentials]:
        """
        Get valid Google credentials.
        
        Returns:
            Optional[Credentials]: Valid credentials or None if authentication fails
        """
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_path):
            with open(self.token_path, 'r') as token:
                creds = Credentials.from_authorized_user_info(
                    json.load(token), self.SCOPES
                )
        
        # Refresh token if expired
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_credentials(creds)
            return creds
        
        # If no valid credentials, start OAuth flow
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path, self.SCOPES
            )
            creds = flow.run_local_server(port=0)
            self._save_credentials(creds)
        
        return creds
    
    def _save_credentials(self, creds: Credentials) -> None:
        """
        Save credentials to token file.
        
        Args:
            creds (Credentials): Credentials to save
        """
        with open(self.token_path, 'w') as token:
            token.write(creds.to_json())
    
    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        return self.get_credentials() is not None
