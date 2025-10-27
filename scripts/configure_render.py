#!/usr/bin/env python3
"""
Render API configuration script.
Programmatically updates Render service configuration.
"""
import os
import sys
import json
import logging
from typing import Dict, Optional, List
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RenderAPIClient:
    """Client for Render.com API."""
    
    BASE_URL = "https://api.render.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Render API client.
        
        Args:
            api_key: Render API key (defaults to RENDER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("RENDER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "RENDER_API_KEY not found. "
                "Set environment variable or pass as argument."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        logger.info(f"{method} {url}")
        
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=data
        )
        
        if response.status_code >= 400:
            logger.error(f"API error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        return response.json() if response.text else {}
    
    def get_service(self, service_id: str) -> Dict:
        """Get service details."""
        return self._request("GET", f"services/{service_id}")
    
    def update_env_vars(
        self,
        service_id: str,
        env_vars: List[Dict[str, str]]
    ) -> Dict:
        """
        Update environment variables for a service.
        
        Args:
            service_id: Render service ID
            env_vars: List of {"key": "VAR_NAME", "value": "var_value"}
        
        Returns:
            Updated service details
        """
        return self._request(
            "PUT",
            f"services/{service_id}/env-vars",
            data=env_vars
        )
    
    def trigger_deploy(self, service_id: str) -> Dict:
        """Trigger a manual deploy."""
        return self._request(
            "POST",
            f"services/{service_id}/deploys",
            data={}
        )
    
    def get_postgres(self, postgres_id: str) -> Dict:
        """Get PostgreSQL instance details."""
        return self._request("GET", f"postgres/{postgres_id}")


def configure_don_research_service(
    service_id: str = "srv-d3qq6o8gjchc73bi0rc0",
    postgres_id: str = "dpg-d3vd9pur433s73cppktg-a",
    api_key: Optional[str] = None
):
    """
    Configure don-research service with database connection.
    
    Args:
        service_id: Render service ID for don-research
        postgres_id: Render PostgreSQL instance ID
        api_key: Render API key (optional)
    """
    client = RenderAPIClient(api_key)
    
    logger.info("=" * 60)
    logger.info("DON Research API - Render Configuration")
    logger.info("=" * 60)
    
    # Step 1: Get PostgreSQL connection details
    logger.info("\n1️⃣  Getting PostgreSQL connection details...")
    postgres = client.get_postgres(postgres_id)
    
    logger.info(f"   Database: {postgres['name']}")
    logger.info(f"   Status: {postgres['status']}")
    logger.info(f"   Region: {postgres['region']}")
    
    # Get internal connection URL (you'll need to get this from dashboard)
    # The API doesn't expose the full connection string for security
    logger.warning(
        "\n⚠️  Note: The Render API doesn't expose database passwords for security."
        "\nYou'll need to get the Internal Database URL from the dashboard:"
        f"\n   https://dashboard.render.com/d/{postgres_id}"
    )
    
    database_url_template = (
        f"postgresql+asyncpg://{postgres['databaseUser']}:[PASSWORD]@"
        f"[HOST]/{postgres['databaseName']}"
    )
    
    logger.info(f"\n   Connection URL template: {database_url_template}")
    logger.info(f"\n   You need to replace [PASSWORD] and [HOST] with actual values from dashboard")
    
    # Step 2: Get current service configuration
    logger.info("\n2️⃣  Getting current service configuration...")
    service = client.get_service(service_id)
    
    logger.info(f"   Service: {service['name']}")
    logger.info(f"   Plan: {service['serviceDetails']['plan']}")
    logger.info(f"   URL: {service['serviceDetails']['url']}")
    
    # Step 3: Prepare environment variables
    logger.info("\n3️⃣  Preparing environment variables...")
    
    # Get DATABASE_URL from user input
    print("\n" + "=" * 60)
    print("DATABASE_URL Configuration")
    print("=" * 60)
    print(f"\nGo to: https://dashboard.render.com/d/{postgres_id}")
    print("\nCopy the 'Internal Database URL' and paste it here:")
    print("(It should look like: postgresql://user:password@host/database)")
    print()
    
    database_url = input("DATABASE_URL: ").strip()
    
    if not database_url:
        logger.error("❌ DATABASE_URL cannot be empty")
        sys.exit(1)
    
    # Convert postgres:// to postgresql+asyncpg://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    env_vars = [
        {
            "key": "DATABASE_URL",
            "value": database_url
        },
        {
            "key": "PYTHON_VERSION",
            "value": "3.11"
        },
        {
            "key": "PORT",
            "value": "8080"
        },
        {
            "key": "DON_STACK_MODE",
            "value": "internal"
        }
    ]
    
    logger.info("\n   Environment variables to set:")
    for var in env_vars:
        if var["key"] == "DATABASE_URL":
            # Mask password in logs
            masked_url = database_url.split("@")[0].split(":")[:-1]
            logger.info(f"   - {var['key']}: {':'.join(masked_url)}:****@...")
        else:
            logger.info(f"   - {var['key']}: {var['value']}")
    
    # Step 4: Update environment variables
    print("\n" + "=" * 60)
    confirm = input("Update environment variables? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        logger.info("❌ Configuration cancelled by user")
        sys.exit(0)
    
    logger.info("\n4️⃣  Updating environment variables...")
    
    try:
        result = client.update_env_vars(service_id, env_vars)
        logger.info("✅ Environment variables updated successfully")
    except Exception as e:
        logger.error(f"❌ Failed to update environment variables: {e}")
        logger.info("\n💡 Alternative: Update manually in dashboard:")
        logger.info(f"   https://dashboard.render.com/web/{service_id}/env-vars")
        sys.exit(1)
    
    # Step 5: Trigger deploy
    logger.info("\n5️⃣  Triggering deployment...")
    print("\nDeploy the service with new configuration? (yes/no): ", end="")
    confirm_deploy = input().strip().lower()
    
    if confirm_deploy == "yes":
        try:
            deploy = client.trigger_deploy(service_id)
            logger.info(f"✅ Deployment triggered: {deploy.get('id', 'N/A')}")
            logger.info(f"\n   Monitor deployment at:")
            logger.info(f"   https://dashboard.render.com/web/{service_id}/deploys")
        except Exception as e:
            logger.error(f"❌ Failed to trigger deployment: {e}")
            logger.info("\n💡 Trigger manually in dashboard:")
            logger.info(f"   https://dashboard.render.com/web/{service_id}")
    else:
        logger.info("⏭️  Skipping deployment (changes saved but not deployed)")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Configuration Complete!")
    logger.info("=" * 60)
    logger.info("\n📋 Next Steps:")
    logger.info("   1. Wait for deployment to complete (~5-8 minutes)")
    logger.info("   2. Check health endpoint:")
    logger.info(f"      curl {service['serviceDetails']['url']}/api/v1/health")
    logger.info("   3. Verify database connection in logs")
    logger.info("   4. Run test suite to validate endpoints")
    logger.info("\n✨ Your DON Research API is ready for production!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Configure DON Research API service on Render"
    )
    parser.add_argument(
        "--service-id",
        default="srv-d3qq6o8gjchc73bi0rc0",
        help="Render service ID"
    )
    parser.add_argument(
        "--postgres-id",
        default="dpg-d3vd9pur433s73cppktg-a",
        help="Render PostgreSQL instance ID"
    )
    parser.add_argument(
        "--api-key",
        help="Render API key (defaults to RENDER_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    try:
        configure_don_research_service(
            service_id=args.service_id,
            postgres_id=args.postgres_id,
            api_key=args.api_key
        )
    except KeyboardInterrupt:
        logger.info("\n\n❌ Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
