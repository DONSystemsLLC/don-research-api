# DON Health Commons License (DHCL) v0.1 - Self-Attestation Form

**Purpose**: This form enables entities to self-declare compliance with DHCL v0.1 Mission-Aligned Entity (MAE) requirements. Completion is required before accessing the Licensed Work (DON Research Node, Real QAC, Signal Sync, Bio Module, etc.).

**Legal Status**: This is a self-attestation, not a legally binding contract. However, materially false statements may result in license termination and legal consequences under applicable fraud statutes.

**Submission**: Email completed form to research@donsystems.com or submit via designated DHCL compliance portal.

---

## Part 1: Entity Information

**Entity Legal Name:**  
_____________________________________________

**Doing Business As (DBA) / Common Name:**  
_____________________________________________

**Entity Type** (check one):
- [ ] Academic Institution (university, research institute, etc.)
- [ ] Public Health Agency (government health department, CDC, WHO, etc.)
- [ ] Non-Profit Organization (501(c)(3) or equivalent)
- [ ] For-Profit Startup (<$50M annual revenue)
- [ ] Clinical Research Organization (hospital, clinic, etc.)
- [ ] Other: _______________________________

**Primary Business Activity:**  
_____________________________________________

**Country of Formation:**  
_____________________________________________

**Principal Address:**  
_____________________________________________
_____________________________________________

**Contact Person (Name & Title):**  
_____________________________________________

**Email:**  
_____________________________________________

**Phone:**  
_____________________________________________

---

## Part 2: Revenue & Ownership (For-Profit Entities Only)

**Annual Gross Revenue (USD):**  
_____________________________________________

**Revenue Source Breakdown:**
- Pharmaceutical sales: $______________
- Biotechnology products: $______________
- Healthcare services: $______________
- Research grants/contracts: $______________
- Other: $______________

**Is your entity majority-owned (>50%) by another company?**
- [ ] No (skip to Part 3)
- [ ] Yes (complete below)

**Parent Company Name:**  
_____________________________________________

**Parent Company Annual Revenue (USD):**  
_____________________________________________

**Parent Company Primary Business:**  
_____________________________________________

**Ownership Percentage:**  
_____________________________________________

---

## Part 3: Prohibited Entity (PE) Status

I hereby attest that (check all that apply):

- [ ] Our entity's annual revenue is **less than $50 million USD**
- [ ] Our entity is **not majority-owned** by a company with >$50M annual pharma/biotech revenue
- [ ] Our entity is **not listed** on the DHCL Designated Exclusion List (DEL)
- [ ] Our entity has **no current plans** to transfer, sublicense, or provide access to the Licensed Work to any Prohibited Entity
- [ ] We understand that **Prohibited Entities** include companies with >$50M annual revenue from pharmaceutical or biotechnology products, as defined in DHCL §1.4

**If any boxes above are UNCHECKED, explain circumstances:**  
_____________________________________________
_____________________________________________

---

## Part 4: Use Case Declaration

**Intended Use of Licensed Work** (check all that apply):
- [ ] Academic research (basic science, publication-focused)
- [ ] Clinical research (patient care improvement, diagnostics)
- [ ] Public health surveillance (disease monitoring, outbreak detection)
- [ ] Drug discovery (target identification, preclinical validation)
- [ ] Computational biology research (algorithms, methods development)
- [ ] Other: _______________________________

**Describe specific research project(s):**  
_____________________________________________
_____________________________________________
_____________________________________________

**Will you be processing Protected Health Information (PHI) or Personally Identifiable Information (PII)?**
- [ ] No
- [ ] Yes (describe data sovereignty measures below)

**Data Sovereignty Measures** (if PHI/PII):  
_____________________________________________
_____________________________________________

---

## Part 5: Compliance Commitments

By signing this form, I attest that our entity will:

### 5.1 Auditability (DHCL §3.1)
- [ ] Maintain audit logs in JSONL, Markdown, or database format
- [ ] Retain audit logs for **minimum 7 years**
- [ ] Include in audit logs: input data provenance, parameter configurations, model versions, output summaries, timestamps
- [ ] Make audit logs available to other MAEs upon reasonable request (excluding PHI/trade secrets)

**Audit Log Storage Method:**  
_____________________________________________

**Audit Log Retention Policy:**  
_____________________________________________

### 5.2 Attribution (DHCL §3.2)
- [ ] Include NOTICE text in all publications, presentations, and distributions
- [ ] Cite original DON Systems work and reference DHCL v0.1-intent license
- [ ] Acknowledge that results are reproducible via audit logs

**Publication Plan** (journals, conferences, preprints):  
_____________________________________________

### 5.3 Data Sovereignty (DHCL §3.3)
- [ ] Use federated deployment patterns (on-site processing, no raw data export to centralized cloud)
- [ ] Keep PHI and PII on originating institution's infrastructure
- [ ] Apply differential privacy or secure multi-party computation if data must be aggregated
- [ ] Prohibit unauthorized transfer of sensitive data to Prohibited Entities

**Data Governance Framework:**  
_____________________________________________

### 5.4 Reciprocity (DHCL §3.4)
- [ ] Contribute meaningful improvements, bug fixes, or extensions back to the Commons
- [ ] Submit contributions within **6 months** of internal use
- [ ] Release contributions under compatible open license (Apache 2.0, MIT, CC-BY, or DHCL itself)
- [ ] Document contributions with methodology, validation results, and reproducibility instructions

**Contribution Mechanism** (GitHub pull request, separate repo, publication):  
_____________________________________________

**Expected Contribution Timeline:**  
_____________________________________________

### 5.5 No Evasion (DHCL §3.5)
- [ ] We will **not** create shell entities or use legal structures to circumvent PE restrictions
- [ ] We will **not** transfer Licensed Work to Prohibited Entities via acquisition, merger, or partnership
- [ ] We will **not** embed Licensed Work in closed proprietary systems without audit trails
- [ ] If our entity changes status (e.g., acquired by PE, revenue exceeds $50M), we will notify DON Systems within 30 days and cease use of Licensed Work

**Change of Status Notification Contact:**  
research@donsystems.com

---

## Part 6: Prohibited Uses (DHCL §4)

I understand and agree that the following uses are **PROHIBITED**:

- [ ] Granting access to Prohibited Entities (companies on DEL with >$50M pharma/biotech revenue)
- [ ] Closed embedding (integrating Licensed Work into proprietary black-box systems without audit trails)
- [ ] Exclusive control (attempting to prevent other MAEs from using Licensed Work via patents, trade secrets, or licensing restrictions)
- [ ] De-identification washing (stripping audit trails or attribution to obscure provenance)

**If any prohibited use is anticipated, explain circumstances:**  
_____________________________________________

---

## Part 7: Commercialization Pathways (DHCL §6)

I understand the following commercialization rules:

### Allowed:
- [ ] Fee-for-service: Running Licensed Work for Prohibited Entities and charging for compute time / expertise
- [ ] Reports to PEs: Providing analysis results, interpretations, visualizations, and recommendations to PEs (without transferring models/algorithms)

### Prohibited:
- [ ] Transferring software, models, or artifacts to Prohibited Entities
- [ ] Sublicensing Licensed Work to Prohibited Entities
- [ ] Allowing Prohibited Entities to self-serve access to Licensed Work

**Anticipated Revenue Model** (if any):  
_____________________________________________

---

## Part 8: Signature & Date

By signing below, I certify that:
1. I am authorized to legally bind the entity named in Part 1
2. All information provided is true and accurate to the best of my knowledge
3. I understand that materially false statements may result in immediate license termination
4. I have read and agree to comply with the full DHCL v0.1 Intent Draft license terms
5. I acknowledge that this is an intent draft (pre-legal review) and final terms may change

**Signatory Name:**  
_____________________________________________

**Title:**  
_____________________________________________

**Signature:**  
_____________________________________________

**Date:**  
_____________________________________________

---

## Submission Instructions

1. **Complete all applicable sections** (skip Part 2 if non-profit/academic)
2. **Sign and date** (digital signatures accepted)
3. **Email to**: research@donsystems.com with subject line "DHCL Self-Attestation: [Entity Name]"
4. **Await confirmation**: DON Systems will review within 5 business days and provide access credentials
5. **Renewals**: Re-attest annually or upon any material change in entity status

---

## Post-Submission: Access & Compliance

**Upon approval, you will receive:**
- API bearer token for DON Research Node endpoints
- Access to Real QAC, Signal Sync, Bio Module, and other licensed tools
- Invitation to MAE community forums (optional)
- Updates on DHCL versioning and DEL additions

**Ongoing Compliance:**
- Maintain audit logs and make available upon request
- Submit reciprocity contributions within 6 months
- Notify DON Systems of any status changes (revenue growth, acquisition, etc.)
- Participate in annual re-attestation process

**Questions?**  
Email research@donsystems.com or open GitHub issue with `[DHCL Compliance]` tag.

---

*Template Version: v0.1-intent (2025-01-24)*  
*Maintained by DON Systems LLC / Foundation*
