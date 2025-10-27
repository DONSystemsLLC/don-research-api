# DON Health Commons License (DHCL) - Changelog

This document tracks the evolution of the DON Health Commons License from intent draft through final ratification.

---

## [v0.1-intent] - 2025-01-24

### Publication Status
- **Phase**: Intent Draft (Pre-Legal Review)
- **RFC Period**: January 24 - February 7, 2025 (14 days)
- **Status**: Open for public comment
- **Next Phase**: v0.2-intent after counsel review

### Intent Draft Content

**Established Framework:**
- Mission-Aligned Entity (MAE) definition: Academic, public health, non-profit, <$50M startup
- Prohibited Entity (PE) definition: >$50M pharma/biotech revenue (DEL registry)
- Grant of Rights: Use, modify, publish results for research purposes
- Core Conditions:
  - Auditability (JSONL/MD/DB audit logs, 7-year retention)
  - Attribution (NOTICE in publications per Appendix A)
  - Data Sovereignty (federated use, PHI stays on-site)
  - Reciprocity (6-month contribution window for improvements)
  - No Evasion (anti-circumvention, no shell entity transfers)

**Prohibitions:**
- PE access to Licensed Work (software/models/artifacts)
- Closed embedding (proprietary black-box integration)
- Exclusive control (preventing other MAEs from using)
- De-identification washing (stripping audit trails)

**Compliance Mechanisms:**
- Self-attestation required for access (Appendix C template)
- Entity type declaration, revenue disclosure, <$50M verification
- Audit log retention commitment (7 years)
- Federated use declaration for PHI data
- Reciprocity timeline acknowledgment

**Commercialization Pathways:**
- ✅ Allowed: Fee-for-service (run models for PEs), reports/interpretations to PEs
- ❌ Prohibited: Model/algorithm transfer to PEs, sublicensing to PEs
- Result: Revenue from expertise, not from locking up instruments

**Termination & Enforcement:**
- Automatic termination on breach
- 30-day cure window for inadvertent violations
- Kill-switch for willful misuse (immediate termination, no cure)
- Survival: Attribution, data sovereignty, reciprocity obligations

**Patent Peace:**
- Defensive patent grant: MAEs receive non-exclusive license to DON patents
- License termination: Automatic if licensee sues over Licensed Work patents
- Exception: Infringement claims within DHCL terms (e.g., PE violation)

**Legal Framework:**
- Warranty Disclaimer: AS-IS provision (no merchantability/fitness guarantees)
- Governing Law: [TBD - likely Massachusetts or international arbitration]
- Severability: Invalid clauses severed, remainder survives
- Entire Agreement: Supersedes prior oral/written agreements

**Appendices:**
- Appendix A: NOTICE template (attribution format for publications)
- Appendix B: Publication process (draft → RFC → pilots → counsel → v1.0)
- Appendix C: Self-attestation checklist (entity type, revenue, compliance)
- Appendix D: DEL registry structure (SHA256 hashes, version control)

### Publication Process (Appendix B)

**Stage 1: Intent Draft (Current)**
- [x] Draft complete with all sections and appendices
- [x] Published to don-research-api repository
- [x] License file created: LICENSE-DHCL-v0.1-intent.md
- [x] NOTICE file created for attribution template
- [x] README.md updated with DHCL reference
- [ ] RFC issue opened for 14-day public comment period
- [ ] Community feedback collection (legal/academic/clinical perspectives)

**Stage 2: Pilot MAE Sign-ons**
- [ ] Recruit 3-5 mission-aligned entities for pilot testing
- [ ] Suggested pilot: Texas A&M University (clinical genomics research)
- [ ] Collect real-world compliance feedback (audit logs, federated use)
- [ ] Validate reciprocity mechanism (6-month contribution window)
- [ ] Test commercialization pathways (fee-for-service to PEs)

**Stage 3: Legal Counsel Review**
- [ ] Engage legal counsel for compliance review (IP, contract, healthcare law)
- [ ] Verify enforceability of MAE/PE restrictions
- [ ] Validate data sovereignty provisions (HIPAA/GDPR compatibility)
- [ ] Review patent peace clause for defensive patent strategy
- [ ] Confirm kill-switch termination provisions
- [ ] Finalize governing law jurisdiction

**Stage 4: v0.2-intent Revision**
- [ ] Incorporate RFC feedback
- [ ] Address counsel concerns
- [ ] Refine MAE/PE definitions based on pilot experiences
- [ ] Adjust revenue threshold ($50M) if needed
- [ ] Update reciprocity window (6 months) based on pilot data
- [ ] Clarify commercialization edge cases

**Stage 5: v1.0 Ratification**
- [ ] Freeze license text (no further changes without community process)
- [ ] Publish v1.0 with counsel sign-off
- [ ] Begin enforcement with DEL registry
- [ ] Establish DHCL governance committee for future amendments
- [ ] Launch public MAE registry (opt-in directory of compliant entities)

### Key Metrics (v0.1-intent)

- **Revenue Threshold**: $50M (startups/small companies qualify as MAEs)
- **Reciprocity Window**: 6 months (contributions must be shared within this period)
- **Audit Log Retention**: 7 years (minimum for reproducibility and compliance)
- **DEL Update Frequency**: Quarterly (version-controlled with SHA256 hashes)

### Known Issues & Questions for RFC

**Entity Classification Edge Cases:**
- Subsidiaries of PEs: Are university spinouts funded by Pfizer/Moderna MAEs?
- Revenue calculation: Gross revenue, net revenue, or research-specific revenue?
- International entities: How to verify revenue for non-US companies?
- Non-profit pharma: Does Médecins Sans Frontières qualify as MAE despite pharma work?

**Compliance Mechanism Questions:**
- Audit log format: Is JSONL/MD/DB sufficiently flexible for diverse use cases?
- Federated use enforcement: How to audit that PHI stayed on-site?
- Reciprocity verification: Who validates that improvements were shared within 6 months?
- Self-attestation honesty: What prevents dishonest self-attestation?

**Commercialization Pathway Ambiguities:**
- Reports to PEs: If report includes code snippets, does that violate "no model transfer"?
- Fee-for-service: Can MAE charge PEs for compute time on proprietary infrastructure?
- Sublicensing gray area: Can MAE grant PEs read-only access to audit logs?

**Termination & Enforcement:**
- Kill-switch invocation: Who has authority to invoke (DON Systems, MAE community, arbitrator)?
- Cure window exceptions: What constitutes "willful misuse" vs. inadvertent violation?
- Post-termination obligations: Do attribution/reciprocity survive termination indefinitely?

**Patent Peace Clause:**
- Defensive patent scope: Does it cover future DON patents filed after license grant?
- Infringement exception: Can MAEs sue PEs who violate DHCL and claim patent infringement?

### Contact & Feedback

**RFC Feedback**: Open GitHub issue with tag `[RFC]` in don-research-api repository  
**Legal Questions**: legal@donsystems.com  
**Pilot Partnerships**: research@donsystems.com  
**General Inquiries**: info@donsystems.com  

---

## Versioning Scheme

- **vX.Y-intent**: Intent drafts (pre-legal review, open for comment)
- **vX.Y-draft**: Post-counsel drafts (legal review complete, pilot testing)
- **vX.Y**: Ratified versions (frozen, no changes without community amendment process)

**Amendment Process (for v1.0+):**
1. Proposal: GitHub issue with `[DHCL Amendment]` tag
2. Discussion: 30-day community comment period
3. Draft: Legal counsel review of proposed changes
4. Vote: MAE community vote (governance committee)
5. Ratification: New version published if 2/3 majority approval

---

*Document maintained by DON Systems LLC / Foundation*  
*Last updated: 2025-01-24*
