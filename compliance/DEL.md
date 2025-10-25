# DON Health Commons License - Designated Exclusion List (DEL)

**Purpose**: This registry identifies Prohibited Entities (PEs) that are explicitly excluded from using the Licensed Work under DHCL v0.1. Entities on this list exceed $50M annual revenue from pharmaceutical or biotechnology products and/or have demonstrated business models incompatible with the mission-aligned Commons.

**Legal Status**: This is a living document subject to quarterly updates. Entities may petition for removal via the dispute resolution process (see DHCL §10).

**Version Control**: Each update includes a SHA256 hash for integrity verification. All historical versions are maintained in this repository.

---

## DEL Schema (JSON)

```json
{
  "del_version": "0.1-intent",
  "effective_date": "2025-01-24",
  "last_updated": "2025-01-24",
  "update_frequency": "quarterly",
  "total_entities": 0,
  "sha256_hash": "",
  "entities": []
}
```

### Entity Entry Schema

```json
{
  "entity_id": "UUID or unique identifier",
  "legal_name": "Full legal entity name",
  "common_name": "DBA / commonly known as",
  "parent_company": "Ultimate parent if subsidiary",
  "revenue_usd": 50000000,
  "revenue_year": 2024,
  "primary_business": "Pharmaceutical | Biotechnology | Healthcare",
  "exclusion_reason": "Revenue threshold | Mission misalignment | Court order",
  "effective_date": "YYYY-MM-DD",
  "dispute_status": "Active | Under Review | Resolved",
  "notes": "Additional context or clarifications",
  "sha256_hash": "Hash of this entity record for verification"
}
```

---

## Current DEL (v0.1-intent)

**Status**: Empty registry during intent draft phase. DEL will be populated during v1.0 ratification after legal counsel review.

**Methodology for Inclusion**:
1. **Automatic Inclusion**: Publicly traded pharma/biotech companies with >$50M annual revenue from drug/diagnostic sales
2. **Discretionary Inclusion**: Private companies with >$50M revenue verified via SEC filings, Bloomberg, or third-party sources
3. **Mission Misalignment**: Entities demonstrating business practices incompatible with public health (e.g., price gouging, blocking generic competition, predatory licensing)
4. **Court Orders**: Entities subject to legal injunctions related to healthcare fraud or DHCL violations

**Data Sources**:
- SEC EDGAR filings (10-K, 10-Q reports)
- Bloomberg terminal revenue data
- PharmaCompass industry databases
- Public financial statements for international entities
- Court records (healthcare fraud, antitrust violations)

---

## Example Entries (Hypothetical - Not Active)

**Note**: The following are illustrative examples for schema documentation. No actual entities are currently on the DEL during intent draft phase.

```json
{
  "del_version": "1.0",
  "effective_date": "2025-06-01",
  "last_updated": "2025-06-01",
  "update_frequency": "quarterly",
  "total_entities": 3,
  "sha256_hash": "abc123...",
  "entities": [
    {
      "entity_id": "PE-2025-001",
      "legal_name": "Example Pharma Corporation",
      "common_name": "ExamplePharma",
      "parent_company": "Example Global Holdings Inc.",
      "revenue_usd": 8500000000,
      "revenue_year": 2024,
      "primary_business": "Pharmaceutical",
      "exclusion_reason": "Revenue threshold (>$50M)",
      "effective_date": "2025-06-01",
      "dispute_status": "None",
      "notes": "Large-cap pharmaceutical company with established drug portfolio. Excluded based on revenue threshold per DHCL §1.4.",
      "sha256_hash": "def456..."
    },
    {
      "entity_id": "PE-2025-002",
      "legal_name": "Example Biotech LLC",
      "common_name": "ExampleBio",
      "parent_company": "None",
      "revenue_usd": 120000000,
      "revenue_year": 2024,
      "primary_business": "Biotechnology",
      "exclusion_reason": "Revenue threshold (>$50M)",
      "effective_date": "2025-06-01",
      "dispute_status": "None",
      "notes": "Mid-size biotech focused on gene therapies. Revenue exceeds MAE threshold.",
      "sha256_hash": "ghi789..."
    },
    {
      "entity_id": "PE-2025-003",
      "legal_name": "Example Diagnostics Inc.",
      "common_name": "ExampleDx",
      "parent_company": "Example Pharma Corporation",
      "revenue_usd": 45000000,
      "revenue_year": 2024,
      "primary_business": "Biotechnology",
      "exclusion_reason": "Parent company on DEL (PE-2025-001)",
      "effective_date": "2025-06-01",
      "dispute_status": "None",
      "notes": "Subsidiary of PE-2025-001. Revenue below threshold but excluded due to parent company status per DHCL §3.5 anti-evasion clause.",
      "sha256_hash": "jkl012..."
    }
  ]
}
```

---

## DEL Update Process

### Quarterly Review Cycle
1. **Data Collection** (Month 1 of each quarter): Scrape SEC filings, Bloomberg data, public financial statements
2. **Verification** (Month 2): Cross-reference revenue data across multiple sources, identify subsidiaries
3. **Draft Update** (Month 3): Prepare new DEL version with added/removed entities
4. **Public Comment** (30 days): Post draft DEL to GitHub, solicit community feedback
5. **Ratification**: Finalize DEL update with SHA256 hash, publish to repository

### Dispute Resolution (DHCL §10)
Entities may petition for removal from DEL if:
- Revenue data was inaccurate (provide corrected financial statements)
- Entity has undergone structural change (spin-off, divestiture, non-profit conversion)
- Mission alignment demonstrated (e.g., policy changes like voluntary drug price caps)

**Petition Process**:
1. Email legal@donsystems.com with subject "DEL Dispute: [Entity Name]"
2. Provide documentation (financial statements, corporate structure, policy changes)
3. DON Systems / DHCL governance committee reviews within 30 days
4. Decision published with rationale (approved / denied)
5. If approved, entity removed from DEL in next quarterly update

---

## SHA256 Hash Verification

Each DEL update includes a SHA256 hash for tamper detection. To verify:

```bash
# Generate hash of DEL JSON file
shasum -a 256 compliance/DEL.json

# Compare against hash in del_version metadata
cat compliance/DEL.json | jq '.sha256_hash'
```

**Version Control**: All historical DEL versions archived in `compliance/del_archive/` with git tags.

---

## DEL Access & API Integration

**Programmatic Access**:
```python
import requests

# Fetch current DEL
del_data = requests.get("https://api.donsystems.com/dhcl/del/latest").json()

# Check if entity is on DEL
def is_prohibited_entity(entity_name: str) -> bool:
    for entity in del_data['entities']:
        if entity_name.lower() in [entity['legal_name'].lower(), entity['common_name'].lower()]:
            return True
    return False
```

**API Endpoints** (planned for v1.0):
- `GET /dhcl/del/latest`: Current DEL version
- `GET /dhcl/del/{version}`: Historical DEL by version
- `GET /dhcl/del/entity/{entity_id}`: Specific entity details
- `POST /dhcl/del/dispute`: Submit dispute petition (requires authentication)

---

## Transparency & Accountability

**Public Audit Trail**:
- All DEL updates committed to public GitHub repository
- Version control history shows additions/removals with rationale
- Community can review and comment on draft updates
- SHA256 hashes prevent tampering

**Governance**:
- DEL maintained by DON Systems during v0.1-intent and v1.0 phases
- Future governance may transition to community committee (MAE representatives)
- Annual review of DEL criteria and revenue thresholds

**Privacy Considerations**:
- DEL includes only publicly available data (SEC filings, public financial statements)
- No personally identifiable information (individual executives, employees)
- Focus on corporate entities, not individuals

---

## Contact & Questions

**DEL Updates**: Subscribe to GitHub repository for automatic notifications  
**Dispute Petitions**: legal@donsystems.com  
**Data Corrections**: research@donsystems.com  
**API Access**: api@donsystems.com  

---

*Document Version: v0.1-intent (2025-01-24)*  
*Next Scheduled Update: 2025-04-01 (Q2 2025)*  
*Maintained by DON Systems LLC / Foundation*
