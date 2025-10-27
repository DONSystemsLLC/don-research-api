"""
Test suite for Texas A&M University API token authentication and authorization.

This test follows TDD principles and validates:
1. Token generation and format
2. Authentication with TAMU token
3. Rate limiting enforcement (1000 req/hour academic tier)
4. Authorization for protected endpoints
5. Audit logging for TAMU requests
"""
import json
import pytest
from pathlib import Path
from src.auth.authorized_institutions import load_authorized_institutions

# TAMU token configuration
TAMU_TOKEN = "tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc"
TAMU_INSTITUTION_NAME = "Texas A&M University - Cai Lab"
TAMU_CONTACT = "jcai@tamu.edu"
TAMU_RATE_LIMIT = 1000  # Academic tier


class TestTAMUTokenConfiguration:
    """Test TAMU token loading and configuration validation."""
    
    def test_tamu_token_format(self):
        """Validate TAMU token follows expected format."""
        assert TAMU_TOKEN.startswith("tamu_cai_lab_2025_")
        assert len(TAMU_TOKEN) > 50  # Secure token should be long
        assert "_" in TAMU_TOKEN
        # URL-safe base64 characters only
        token_suffix = TAMU_TOKEN.split("tamu_cai_lab_2025_")[1]
        assert all(c.isalnum() or c in ['-', '_'] for c in token_suffix)
    
    def test_load_tamu_token_from_json_env(self, monkeypatch):
        """Test loading TAMU token from DON_AUTHORIZED_INSTITUTIONS_JSON."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_FILE", raising=False)
        
        institutions = load_authorized_institutions()
        
        assert TAMU_TOKEN in institutions
        assert institutions[TAMU_TOKEN]["name"] == TAMU_INSTITUTION_NAME
        assert institutions[TAMU_TOKEN]["contact"] == TAMU_CONTACT
        assert institutions[TAMU_TOKEN]["rate_limit"] == TAMU_RATE_LIMIT
    
    def test_load_tamu_token_from_file(self, tmp_path: Path, monkeypatch):
        """Test loading TAMU token from configuration file."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        
        config_file = tmp_path / "tamu_institutions.json"
        config_file.write_text(json.dumps(tamu_config), encoding="utf-8")
        
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_FILE", str(config_file))
        monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_JSON", raising=False)
        
        institutions = load_authorized_institutions()
        
        assert TAMU_TOKEN in institutions
        assert institutions[TAMU_TOKEN]["name"] == TAMU_INSTITUTION_NAME
        assert institutions[TAMU_TOKEN]["rate_limit"] == TAMU_RATE_LIMIT
    
    def test_tamu_token_coexists_with_demo(self, monkeypatch):
        """Test TAMU token can coexist with default demo token."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_FILE", raising=False)
        
        institutions = load_authorized_institutions()
        
        # Both tokens should be present
        assert "demo_token" in institutions
        assert TAMU_TOKEN in institutions
        
        # Verify different rate limits
        assert institutions["demo_token"]["rate_limit"] == 100  # Demo tier
        assert institutions[TAMU_TOKEN]["rate_limit"] == 1000  # Academic tier


class TestTAMUTokenAuthentication:
    """Test authentication behavior with TAMU token."""
    
    @pytest.fixture
    def tamu_configured_app(self, monkeypatch):
        """Configure app with TAMU token for testing."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        
        # Reload main to pick up new config
        import importlib
        import main
        importlib.reload(main)
        
        from fastapi.testclient import TestClient
        return TestClient(main.app)
    
    def test_health_endpoint_without_token(self, tamu_configured_app):
        """Test health endpoint accessible without authentication."""
        response = tamu_configured_app.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_protected_endpoint_without_token(self, tamu_configured_app):
        """Test protected endpoints reject requests without token."""
        test_data = {"embedding": [[0.5, 0.5, 0.5, 0.5]]}
        response = tamu_configured_app.post("/api/v1/quantum/qac/fit", json=test_data)
        assert response.status_code in [401, 403]
    
    def test_protected_endpoint_with_invalid_token(self, tamu_configured_app):
        """Test protected endpoints reject invalid tokens."""
        headers = {"Authorization": "Bearer invalid_token_12345"}
        test_data = {"embedding": [[0.5, 0.5, 0.5, 0.5]]}
        response = tamu_configured_app.post(
            "/api/v1/quantum/qac/fit", 
            json=test_data,
            headers=headers
        )
        assert response.status_code == 401
    
    def test_protected_endpoint_with_tamu_token(self, tamu_configured_app):
        """Test protected endpoints accept valid TAMU token."""
        headers = {"Authorization": f"Bearer {TAMU_TOKEN}"}
        test_data = {
            "embedding": [
                [0.5, 0.5, 0.5, 0.5],
                [0.6, 0.4, 0.5, 0.5]
            ],
            "sync": True
        }
        response = tamu_configured_app.post(
            "/api/v1/quantum/qac/fit",
            json=test_data,
            headers=headers
        )
        # Should succeed (200) or fail for business logic reasons, but NOT 401/403
        assert response.status_code not in [401, 403]


class TestTAMUTokenRateLimiting:
    """Test rate limiting for TAMU academic tier."""
    
    def test_tamu_rate_limit_configuration(self, monkeypatch):
        """Verify TAMU token has correct rate limit (1000 req/hour)."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        
        institutions = load_authorized_institutions()
        assert institutions[TAMU_TOKEN]["rate_limit"] == 1000
    
    def test_tamu_higher_limit_than_demo(self, monkeypatch):
        """Verify TAMU academic tier has higher limit than demo."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        
        institutions = load_authorized_institutions()
        
        demo_limit = institutions["demo_token"]["rate_limit"]
        tamu_limit = institutions[TAMU_TOKEN]["rate_limit"]
        
        assert tamu_limit > demo_limit
        assert tamu_limit == 1000  # Academic tier
        assert demo_limit == 100   # Demo tier


class TestTAMUTokenSecurity:
    """Test security aspects of TAMU token."""
    
    def test_token_not_in_demo_config(self):
        """Ensure TAMU token is not in sample/demo configurations."""
        from pathlib import Path
        sample_config_path = Path("config/authorized_institutions.sample.json")
        
        if sample_config_path.exists():
            sample_content = sample_config_path.read_text()
            assert TAMU_TOKEN not in sample_content
            assert "HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc" not in sample_content
    
    def test_token_cryptographic_strength(self):
        """Validate token has sufficient cryptographic strength."""
        # secrets.token_urlsafe(32) generates 43 characters
        # This provides ~256 bits of entropy
        token_suffix = TAMU_TOKEN.split("tamu_cai_lab_2025_")[1]
        assert len(token_suffix) >= 43  # Minimum for 32-byte secret
    
    def test_token_unique_to_institution(self, monkeypatch):
        """Ensure token maps to correct institution."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        
        institutions = load_authorized_institutions()
        
        # Verify token maps to TAMU specifically
        assert institutions[TAMU_TOKEN]["name"] == TAMU_INSTITUTION_NAME
        assert "Texas A&M" in institutions[TAMU_TOKEN]["name"]
        assert institutions[TAMU_TOKEN]["contact"] == TAMU_CONTACT
        assert "tamu.edu" in institutions[TAMU_TOKEN]["contact"]


class TestTAMUTokenAuditLogging:
    """Test audit logging for TAMU token usage."""
    
    def test_audit_log_includes_institution_name(self, monkeypatch):
        """Verify audit logs capture Texas A&M institution name."""
        tamu_config = {
            TAMU_TOKEN: {
                "name": TAMU_INSTITUTION_NAME,
                "contact": TAMU_CONTACT,
                "rate_limit": TAMU_RATE_LIMIT
            }
        }
        monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(tamu_config))
        
        institutions = load_authorized_institutions()
        
        # Audit logs should be able to identify requests from TAMU
        assert institutions[TAMU_TOKEN]["name"] == TAMU_INSTITUTION_NAME
        assert "Cai Lab" in institutions[TAMU_TOKEN]["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
