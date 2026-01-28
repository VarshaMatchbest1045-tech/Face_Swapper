
import requests
from typing import Dict
import os
import math
from dotenv import load_dotenv

# Explicitly load .env file
load_dotenv()

# Load from environment variables (assumes these are loaded elsewhere or accessible)
# API users should set these in their .env file or environment
CREDIT_SERVICE_BASE_URL = os.getenv("CREDIT_SERVICE_BASE_URL", "https://api.xelta.ai")
INTERNAL_SERVICE_KEY = os.getenv("INTERNAL_SERVICE_KEY", "")

def _get_headers() -> Dict[str, str]:
    """Get headers required for Credit Service API."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "internal-credit-client/1.0",
    }
    
    if not INTERNAL_SERVICE_KEY:
        # We might want to log a warning or raise an error if key is missing when strictly required
        # For now, we'll proceed but the API call will likely fail 401/403
        pass
    
    headers["x-internal-key"] = INTERNAL_SERVICE_KEY.strip()
    return headers

def get_user_balance(user_id: str) -> Dict:
    """Get user credit balance."""
    url = f"{CREDIT_SERVICE_BASE_URL.rstrip('/')}/service/users/credit-balance"
    
    # Ensure key is present
    if not INTERNAL_SERVICE_KEY:
         raise RuntimeError("INTERNAL_SERVICE_KEY is not set in environment variables")

    response = requests.get(
        url,
        params={"userId": user_id},
        headers=_get_headers(),
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def deduct_credits(user_id: str, amount: int, resource_type: str, resource_id: str) -> Dict:
    """Deduct credits from user account."""
    url = f"{CREDIT_SERVICE_BASE_URL.rstrip('/')}/service/users/credits-debits"
    
    if not INTERNAL_SERVICE_KEY:
         raise RuntimeError("INTERNAL_SERVICE_KEY is not set in environment variables")
         
    payload = {
        "type": "USAGE",
        "userId": user_id,
        "amount": -abs(amount),  # Must be negative
        "resourceType": resource_type,
        "resourceId": resource_id
    }
    response = requests.post(
        url,
        json=payload,
        headers=_get_headers(),
        timeout=30
    )
    response.raise_for_status()
    return response.json()
