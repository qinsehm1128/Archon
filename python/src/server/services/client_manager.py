"""
Client Manager Service

Manages database and API client connections.
"""

import os
import re

from supabase import Client, create_client

from ..config.logfire_config import search_logger


def get_supabase_client() -> Client:
    """
    Get a Supabase client instance.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
        )

    try:
        # Create Supabase client with SSL verification disabled
        import httpx
        from supabase.client import ClientOptions
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # Create custom httpx client with SSL verification disabled
        http_client = httpx.Client(verify=False, timeout=30.0)

        # Create Supabase client with custom HTTP client
        client_options = ClientOptions()
        client_options.custom_http_client = http_client

        client = create_client(url, key, options=client_options)

        # Extract project ID from URL for logging purposes only
        match = re.match(r"https://([^.]+)\.supabase\.co", url)
        if match:
            project_id = match.group(1)
            search_logger.debug(f"Supabase client initialized - project_id={project_id}")

        return client
    except Exception as e:
        search_logger.error(f"Failed to create Supabase client: {e}")
        raise
