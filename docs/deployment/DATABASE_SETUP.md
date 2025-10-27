# Database Setup Guide

## Issue: Local Migration Failures

We're encountering **"SSL connection has been closed unexpectedly"** errors when trying to run migrations from a local machine to Render PostgreSQL. This is likely due to:

1. **Render's network security**: Free/Starter tier PostgreSQL databases may restrict external connections
2. **Firewall rules**: Your local network or Render's firewall may be blocking the connection
3. **SSL handshake issues**: Connection drops during SSL negotiation

## Solution: Run Migrations from Render Environment

Since Render services can connect to their PostgreSQL databases using internal networking, we should run migrations from within the deployed service.

### Option 1: Using Render Shell (Recommended)

1. Go to your Render dashboard: https://dashboard.render.com
2. Navigate to your web service: `don-research-api`
3. Click on the **"Shell"** tab
4. Run the migration command:
   ```bash
   python run_migrations.py
   ```

This script will:
- Check for the DATABASE_URL environment variable
- Run `alembic upgrade head`
- Report success or failure

### Option 2: Manual Migration via Shell

If the script doesn't work, you can run Alembic directly in the Render shell:

```bash
# Set DATABASE_URL if not already set
export DATABASE_URL="<your-internal-database-url>"

# Run migrations
alembic upgrade head

# Verify migrations
alembic current
```

### Option 3: Add Migration to Deployment Process

Modify your `render.yaml` or add a build command:

```yaml
services:
  - type: web
    name: don-research-api
    env: python
    buildCommand: |
      pip install -r requirements.txt
      python run_migrations.py
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

This will automatically run migrations on every deployment.

## Database Connection Details

### External URL (for local development - if accessible)
```
postgresql://don_research_postgres_user:7LTvWaDivxWiOaqlvRkY86sIRLAK6CJ5@dpg-d3vd9pur433s73cppktg-a.oregon-postgres.render.com:5432/don_research_postgres
```

### Internal URL (for Render services)
Check your Render database dashboard for the **Internal Database URL**. It will look like:
```
postgresql://don_research_postgres_user:...@dpg-d3vd9pur433s73cppktg-a:5432/don_research_postgres
```

Note: The internal URL uses the shorter hostname without the `.oregon-postgres.render.com` suffix.

## Environment Variables to Set

In your Render service settings, add:

```
DATABASE_URL=<your-internal-or-external-database-url>
```

The application will automatically convert it to the async format (`postgresql+asyncpg://...`) as needed.

## Verification Steps

After running migrations, verify with:

```bash
# Check current migration version
alembic current

# Should output: 001_initial (head)

# Or connect to the database and check tables
psql $DATABASE_URL -c "\dt"

# Should show tables: qac_models, vector_stores, jobs, usage_logs, audit_logs
```

## Troubleshooting

### "SSL/TLS required" error
- Ensure `sslmode=require` is in the connection string OR
- The application is using SSL context automatically

### "Connection closed unexpectedly"
- Use the **Internal Database URL** from Render dashboard
- Run migrations from within Render environment (Shell or build command)

### "No module named 'greenlet'"
- Ensure `greenlet` is in `requirements.txt` (already added)
- Run `pip install -r requirements.txt`

### "metadata column conflict"
- Already fixed in models.py (renamed to `model_metadata`, `vector_metadata`)
- Clear Python cache if needed: `find . -type d -name __pycache__ -exec rm -rf {} +`

## Migration File Details

The migration creates:

1. **qac_models** table - Quantum error correction model storage
2. **vector_stores** table - Compressed genomics vectors with pgvector
3. **jobs** table - Async job tracking
4. **usage_logs** table - API usage aggregation
5. **audit_logs** table - Detailed request/response audit trail

All tables include:
- Soft delete support (`deleted_at`, `is_deleted`)
- Timestamp tracking (`created_at`, `updated_at`)
- Appropriate indexes for query performance
