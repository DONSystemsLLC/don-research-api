#!/usr/bin/env python3
"""
Run database migrations from within the deployed environment.
This script is designed to run on Render where the database connection is local.
"""
import os
import sys
import subprocess

def main():
    """Run Alembic migrations."""
    # Ensure we have a DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)
    
    print(f"✅ Found DATABASE_URL: {database_url[:30]}...")
    
    # Run Alembic upgrade
    print("🔄 Running database migrations...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Migrations completed successfully!")
        print(result.stdout)
    else:
        print("❌ Migration failed!")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
