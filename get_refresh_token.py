import os
import requests
from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load your environment variables
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTH_URI = os.getenv("GOOGLE_AUTH_URI")
TOKEN_URI = os.getenv("GOOGLE_TOKEN_URI")
SCOPE = "https://www.googleapis.com/auth/adwords"

# Validate required environment variables
if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, AUTH_URI, TOKEN_URI]):
    missing_vars = []
    if not CLIENT_ID: missing_vars.append("GOOGLE_CLIENT_ID")
    if not CLIENT_SECRET: missing_vars.append("GOOGLE_CLIENT_SECRET")
    if not REDIRECT_URI: missing_vars.append("REDIRECT_URI")
    if not AUTH_URI: missing_vars.append("GOOGLE_AUTH_URI")
    if not TOKEN_URI: missing_vars.append("GOOGLE_TOKEN_URI")
    
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Step 1: Direct user to this URL
params = {
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": SCOPE,
    "access_type": "offline",
    "prompt": "consent"
}
auth_url = f"{AUTH_URI}?{urlencode(params)}"

print("🔗 Open the following URL in your browser and grant access:")
print(auth_url)

# Step 2: Start temporary server to catch the redirect
class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if "code=" in self.path:
            code = self.path.split("code=")[-1].split("&")[0]

            # Exchange code for refresh token
            data = {
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            response = requests.post(TOKEN_URI, data=data)
            tokens = response.json()

            refresh_token = tokens.get("refresh_token")
            access_token = tokens.get("access_token")

            if refresh_token:
                print(f"\n✅ Refresh Token:\n{refresh_token}")
                print(f"\n✅ Access Token:\n{access_token}")
            else:
                print("\n❌ No refresh token returned. Try again with prompt=consent.")

            # Show confirmation in browser
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h1>You may now close this window.</h1>")

            # Shut down server
            self.server.shutdown()

# Start the server and wait for the code
server = HTTPServer(("localhost", 8080), OAuthCallbackHandler)
server.serve_forever()
