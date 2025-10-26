# API Token Provisioning

**Audience:** DON Systems engineering & operations

## Overview

The Research API now loads authorized institutions from either:

1. `DON_AUTHORIZED_INSTITUTIONS_JSON` – inline JSON string for ad-hoc overrides.
2. `DON_AUTHORIZED_INSTITUTIONS_FILE` – path to a JSON file containing the same structure.

If neither is provided, the service falls back to the baked-in demo token only.

## Onboarding a New Institution (e.g., Texas A&M Cai Lab)

1. **Generate a secure token**

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Record the output securely. Example format: `tamu_cai_lab_2025_FooBar...`.

2. **Create the JSON payload**

   ```json
   {
     "tamu_cai_lab_2025_FooBar": {
       "name": "Texas A&M | Cai Lab",
       "contact": "jcai@tamu.edu",
       "rate_limit": 1000
     }
   }
   ```


3. **Deploy the token**
   - Preferred: store the JSON in a secrets manager (Render secret file, AWS SSM, etc.) and point `DON_AUTHORIZED_INSTITUTIONS_FILE` to the mounted path.
   - Alternate: set `DON_AUTHORIZED_INSTITUTIONS_JSON` directly in the service environment.

4. **Restart the API service** so the loader picks up the new configuration.

5. **Distribute the token** to the partner via the secure channel defined in the NDA.

6. **Verify access** by calling `/api/v1/health` with the new token and confirming a `200` response.

## Notes

- Tokens are opaque strings; no hashing is performed. Rotate them if compromise is suspected.
- Rate limits remain per-token. Increase the `rate_limit` field if the partner requires higher throughput.
- The configuration loader validates each entry (non-empty `name`, `contact`, and positive `rate_limit`). Invalid payloads cause startup failures, surfacing in logs.
- Keep `config/authorized_institutions.sample.json` as a reference template; do not commit real tokens to the repository.
