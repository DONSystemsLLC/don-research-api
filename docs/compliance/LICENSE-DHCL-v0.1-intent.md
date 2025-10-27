# DON Health Commons License (DHCL) v0.1 — Intent Draft

> **Status:** Intent draft for review and public comment. Not legal advice. Final terms subject to counsel review and jurisdictional tailoring.

## 0. Preamble & Purpose

The DON Health Commons License ("DHCL") enables **mission‑aligned** clinical and academic research to use the DON Research Node (including Real QAC, Signal Sync, parasite cleanup, evolution reports, adapters, and artifacts) while preventing enclosure by actors whose incentives conflict with public health. DHCL's goals are to:

1. **Accelerate reproducible medical research** by making coherence‑preserving tools broadly available to non‑extractive institutions.
2. **Protect the instrument** (algorithms, models, and methods) from being absorbed into closed, profit‑maximizing pipelines.
3. **Guarantee auditability and data sovereignty** via mandatory logging and federated usage patterns.

## 1. Key Definitions

**Software**: The DON Research Node code, QAC engines, analysis modules, adapters, schemas, and reference implementations released under DHCL.

**Model**: Any trained or fitted artifact produced by the Software (e.g., QAC model graphs, Laplacians, embeddings, or stabilization weights), including derived parameters.

**Artifacts**: Output files (collapse maps, vectors, reports, logs) created by the Software or Models.

**Licensed Work**: Software + Models + Artifacts, except where third‑party licenses supersede.

**Mission‑Aligned Entity (MAE)**: (a) accredited academic institutions; (b) public hospitals and public health agencies; (c) non‑profit research organizations; (d) startups with < **$50M** annual revenue and not majority‑owned by a Prohibited Entity.

**Prohibited Entity (PE)**: (i) companies on the **Designated Exclusion List (DEL)** published by DON Systems Foundation (e.g., top‑tier pharma/biotech with > **$50M** annual biotech/pharma revenue); (ii) any entity majority‑owned by a PE; (iii) integrators embedding the Licensed Work into closed, non‑auditable decision systems.

**Federated Use**: Workflows where raw patient‑level data remains on‑site; only Models, metrics, and non‑PHI summaries are exchanged.

**Audit Logs**: JSONL/Markdown/DB records of seeds, engines, caps, health snapshots, and inputs/outputs sufficient to reproduce results.

## 2. Grant of Rights

Subject to compliance with this License, DON Systems LLC (or its Foundation) grants MAEs a worldwide, royalty‑free, non‑exclusive, non‑transferable, revocable license to:

* Use, reproduce, and modify the Software and Models;
* Generate and use Artifacts; and
* Publish results, benchmarks, and derived scientific works,

**solely** for research, clinical validation, and non‑extractive commercialization under Section 6.

## 3. Conditions (You Must)

1. **Auditability**: Maintain and, upon request from collaborators or regulators, provide Audit Logs for any published results derived from the Licensed Work.
2. **Attribution**: Include the NOTICE (Appendix A) in publications and software distributions.
3. **Data Sovereignty**: When processing PHI/PII, use Federated Use patterns. Do not upload raw patient‑level data to third parties unless independently authorized.
4. **Reciprocity (Time‑boxed)**: Improvements to core algorithms or reproducibility features **must** be contributed back under DHCL within **6 months** of production use or publication.
5. **No Evasion**: Do not route through affiliates or resellers to bypass Prohibited Entity restrictions.

## 4. Prohibitions (You May Not)

* **PE Access**: Provide the Software, Models, or integration interfaces to any PE or for the primary benefit of a PE.
* **Closed Embedding**: Embed the Licensed Work into non‑auditable, proprietary decision systems without exposing logs and parameters needed for independent verification.
* **Exclusive Control**: Assert patents or contracts that would prevent other MAEs from using the Licensed Work as permitted here.
* **De‑Identification Washing**: Claim non‑PHI status while retaining re‑identification keys; all federated exchanges must be genuinely non‑PHI.

## 5. Compliance & Verification

Licensee will adopt reasonable technical and organizational measures to ensure compliance (access controls, logging, retention). The Licensor may request a **compliance self‑attestation** (template provided) and may designate a neutral auditor in case of credible breach allegations; audits will be limited in scope to verifying compliance with this License and confidentiality will be preserved.

## 6. Commercialization Pathways

**Allowed for MAEs**:

* Fee‑for‑service analyses ("results‑only"), publications, and SaaS access for other MAEs.
* Sale of **reports** and **diagnostic/companion** outputs to PEs, **provided** no Software, Models, adjacency graphs, or stabilization weights are transferred.

**Restricted**: Any sublicensing or technology transfer to PEs requires a separate DON Systems commercial license with explicit Commons protections.

## 7. Termination

This License terminates automatically if Licensee materially breaches Sections 3–4. Upon termination, Licensee must cease use and distribution of the Licensed Work and destroy confidential artifacts. Good‑faith cure within **30 days** may reinstate the License at Licensor's sole discretion.

**Kill‑Switch Clause**: In case of willful misuse (e.g., PE enclosure, fraud), Licensor may immediately revoke access keys and publish the revocation in the DEL registry.

## 8. Patent Peace

Licensee grants a defensive, royalty‑free patent license to Licensor and all MAEs for claims necessarily infringed by contributions to the Licensed Work. If Licensee initiates a patent suit alleging that the Licensed Work infringes its patents, this License terminates immediately for that Licensee.

## 9. Warranty & Liability

THE LICENSED WORK IS PROVIDED "AS IS" WITHOUT WARRANTY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, LICENSOR SHALL NOT BE LIABLE FOR ANY INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS LICENSE OR USE OF THE LICENSED WORK.

## 10. Governing Law & Dispute Resolution

To be defined with counsel. Recommended: venue where DON Systems Foundation is organized; mediation followed by binding arbitration; public‑interest injunctive relief available to preserve Commons integrity.

## 11. Versioning & Updates

This is **DHCL v0.1 (Intent)**. Licensor may publish updates (v0.2‑intent, v1.0). Material changes will include a public comment period (see Appendix B).

---

## Appendix A — NOTICE Template

```
This work uses the DON Research Node under the DON Health Commons License (DHCL) v0.1‑intent.
© DON Systems LLC / Foundation. https://don‑systems.org/licenses/dhcl
Coherence‑critical results were produced with Real QAC; seeds, engines, caps, and health snapshots
are retained in audit logs per DHCL §3.
```

## Appendix B — Publication & Comment Process (to "Seal" v0.1‑intent)

1. Publish this draft in a public repo (`LICENSE‑DHCL‑v0.1‑intent.md`, `NOTICE`, `CHANGELOG`), tag `dhcl‑v0.1‑intent`.
2. Open an **RFC issue** with a **14‑day** public comment window.
3. Collect sign‑ons from pilot MAEs (e.g., TAMU) as *implementers*.
4. Send to counsel with: intent, prohibited‑entity rationale, export/privacy notes, and sample use‑cases.
5. Incorporate counsel edits; publish **v0.2‑intent**.
6. When pilots commence, freeze **v1.0**; add jurisdictional riders if needed.

## Appendix C — Compliance Self‑Attestation (excerpt)

* Entity type and revenue threshold (< $50M) ✅/❌
* No transfer to PEs or affiliates ✅/❌
* Audit Logs retained for 7 years ✅/❌
* Federated Use (no PHI sharing) ✅/❌
* Reciprocity timeline (< 6 months) ✅/❌

## Appendix D — Designated Exclusion List (DEL)

Maintained as a signed JSON file with SHA256 digest. Includes parent entities and majority‑owned subsidiaries. Updates are versioned; disputes resolved via Section 10.

---

### Human‑readable Summary (non‑binding)

Use the DON instrument to make medicine reproducible and portable—**if you're aligned with public health**. Share improvements, keep logs, protect patient data, and don't hand the instrument to companies that will lock it up. Share **results** with anyone; the instrument stays in the Commons.
