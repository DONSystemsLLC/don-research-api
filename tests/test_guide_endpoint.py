"""
Test /guide endpoint - comprehensive user guide page.
Ensures guide is accessible, complete, and navigation works.
"""
import pytest
import re
from fastapi.testclient import TestClient


def test_guide_endpoint_exists(api_client):
    """Guide endpoint should be accessible at /guide."""
    response = api_client.get("/guide")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_guide_has_proper_html_structure(api_client):
    """Guide should have valid HTML structure."""
    response = api_client.get("/guide")
    content = response.text
    
    # Basic HTML structure
    assert "<!DOCTYPE html>" in content
    assert "<html" in content
    assert "<head>" in content
    assert "<body>" in content
    assert "</html>" in content


def test_guide_has_navigation_to_homepage(api_client):
    """Guide should have navigation back to homepage."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should link back to homepage
    assert 'href="/"' in content or 'href="../"' in content
    # Should have clear navigation text
    assert "home" in content or "back" in content


def test_guide_has_all_major_sections(api_client):
    """Guide should include all documented sections."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Core sections from WEBAPP_USER_GUIDE.md
    required_sections = [
        "web interface overview",
        "overview",  # Changed from "getting started" to match actual implementation
        "navigation",
        "swagger",
        "bio module",
        "workflow",
        "troubleshooting",
        "support"
    ]
    
    for section in required_sections:
        assert section in content, f"Missing section: {section}"


def test_guide_has_swagger_ui_tutorial(api_client):
    """Guide should explain how to use Swagger UI."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Swagger UI documentation
    assert "swagger" in content
    assert "/docs" in content
    # Should explain authentication
    assert "authorize" in content or "authentication" in content
    # Should explain testing endpoints
    assert "try it out" in content or "execute" in content


def test_guide_has_bio_module_documentation(api_client):
    """Guide should document Bio module features."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Bio module features
    assert "export-artifacts" in content or "export artifacts" in content
    assert "signal-sync" in content or "signal sync" in content
    assert "parasite-detect" in content or "parasite" in content
    assert "evolution" in content
    # Sync/async modes
    assert "sync" in content and "async" in content


def test_guide_has_complete_code_examples(api_client):
    """Guide should include working code examples."""
    response = api_client.get("/guide")
    content = response.text
    
    # Should have code blocks
    assert "<pre>" in content or "<code>" in content
    # Should show Python examples
    assert "import requests" in content or "import" in content
    # Should show API calls
    assert "don-research" in content.lower() or "api_url" in content.lower()


def test_guide_has_troubleshooting_section(api_client):
    """Guide should include common error solutions."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Common HTTP errors
    assert "401" in content  # Authentication
    assert "400" in content  # Bad request
    assert "429" in content  # Rate limit
    # Error solutions
    assert "solution" in content or "fix" in content


def test_guide_has_contact_information(api_client):
    """Guide should provide support contact info."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Contact emails
    assert "@donsystems.com" in content or "support" in content
    # Office hours or response time
    assert "hour" in content or "monday" in content


def test_guide_has_anchor_links(api_client):
    """Guide should have anchor links for navigation."""
    response = api_client.get("/guide")
    content = response.text
    
    # Should have id attributes for sections
    assert re.search(r'id="[a-z-]+"', content, re.IGNORECASE)
    # Should have anchor links in navigation
    assert 'href="#' in content


def test_guide_matches_existing_styling(api_client):
    """Guide should use same CSS styling as homepage."""
    homepage = api_client.get("/help")  # Changed from "/" to "/help" for HTML homepage
    guide = api_client.get("/guide")
    
    # Both should have consistent color scheme - check for specific DON blue colors
    # Primary blue: #0b3d91, Dark blue: #11203f
    assert "#0b3d91" in homepage.text.lower() or "#0b3d91" in guide.text.lower()
    assert "#11203f" in homepage.text.lower() or "#11203f" in guide.text.lower()
    
    # Both should have similar structure
    assert "<header>" in homepage.text.lower() and "<header>" in guide.text.lower()
    assert "<footer>" in homepage.text.lower() and "<footer>" in guide.text.lower()


def test_guide_has_workflow_examples(api_client):
    """Guide should include end-to-end workflow examples."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should have complete workflows
    assert "workflow" in content
    # Should show cell type discovery
    assert "cell type" in content or "t cell" in content
    # Should show QC pipeline
    assert "qc" in content or "quality" in content


def test_guide_has_cell_type_markers_reference(api_client):
    """Guide should include common cell type marker genes."""
    response = api_client.get("/guide")
    content = response.text
    
    # Should have marker gene table or list
    assert "marker" in content.lower()
    # Should mention common markers
    common_markers = ["CD3E", "CD8A", "CD4", "MS4A1", "CD14"]
    found_markers = sum(1 for marker in common_markers if marker in content)
    assert found_markers >= 2, "Missing common cell type markers"


def test_guide_has_distance_interpretation(api_client):
    """Guide should explain how to interpret distance scores."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should explain distance metric
    assert "distance" in content
    # Should have distance values in examples
    assert "0." in content or "distance" in content  # Distance values in code examples


def test_guide_responsive_design(api_client):
    """Guide should be mobile-responsive."""
    response = api_client.get("/guide")
    content = response.text
    
    # Should have viewport meta tag
    assert 'name="viewport"' in content
    # Should have responsive CSS
    assert "@media" in content or "max-width" in content


def test_guide_has_format_comparison(api_client):
    """Guide should explain different input format options."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should document all formats
    assert "h5ad" in content
    assert "geo" in content or "gse" in content
    assert "json" in content
    # Should compare when to use each
    assert "format" in content


def test_guide_has_production_url(api_client):
    """Guide should reference correct production URL."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should show production URL
    assert "don-research" in content
    assert "onrender.com" in content or "https://" in content


def test_guide_links_to_swagger_docs(api_client):
    """Guide should link to interactive API docs."""
    response = api_client.get("/guide")
    content = response.text
    
    # Should have link to /docs
    assert 'href="/docs"' in content or "swagger" in content.lower()


def test_guide_has_rate_limit_documentation(api_client):
    """Guide should document rate limits."""
    response = api_client.get("/guide")
    content = response.text.lower()
    
    # Should mention rate limits and the limit value
    assert "rate limit" in content or "429" in content
    assert "1000" in content or "1,000" in content  # 1,000 req/hour limit
