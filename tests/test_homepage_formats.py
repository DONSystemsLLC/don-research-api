"""
Test homepage documentation accuracy for supported data formats.
Ensures all mentioned formats are actually supported by the API.
"""
import pytest
import re
from fastapi.testclient import TestClient


def test_homepage_mentions_h5ad_format(api_client):
    """Homepage should document H5AD file support."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention h5ad format
    assert ".h5ad" in content
    assert "anndata" in content or "single-cell" in content


def test_homepage_mentions_geo_accessions(api_client):
    """Homepage should document GEO accession support."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention GEO accessions
    assert "geo" in content or "gse" in content
    # Should show example format
    assert re.search(r"gse\d+", content, re.IGNORECASE)


def test_homepage_mentions_url_downloads(api_client):
    """Homepage should document direct URL downloads."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention URL/HTTP support
    assert "url" in content or "http" in content or "download" in content


def test_homepage_mentions_gene_list_queries(api_client):
    """Homepage should document gene list query support."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention gene lists
    assert "gene" in content
    assert "list" in content or "array" in content
    # Should show JSON format
    assert "json" in content


def test_homepage_mentions_text_queries(api_client):
    """Homepage should document text query support."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention text/natural language queries
    assert "text" in content or "query" in content
    assert "cell type" in content or "tissue" in content


def test_homepage_has_data_format_table(api_client):
    """Homepage should have a comparison table of supported formats."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text
    
    # Should have table structure
    assert "<table" in content
    assert "<th" in content
    assert "<td" in content


def test_homepage_links_to_guide(api_client):
    """Homepage should link to detailed guide."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should have link to /guide
    assert "/guide" in content or "user guide" in content or "detailed guide" in content


def test_homepage_bio_module_format_requirements(api_client):
    """Homepage should clarify Bio module format requirements."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text.lower()
    
    # Should mention Bio module format restrictions
    assert "bio" in content
    # Should indicate H5AD requirement for Bio endpoints
    if "bio" in content:
        # Bio sections should mention h5ad requirement
        assert ".h5ad" in content


def test_homepage_has_format_examples(api_client):
    """Homepage should provide examples for each format."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text
    
    # Should have code examples
    assert "<pre>" in content or "<code>" in content
    # Should show example filenames/accessions
    assert re.search(r'\.h5ad|GSE\d+|".*?"', content, re.IGNORECASE)


def test_homepage_responsive_design(api_client):
    """Homepage should have responsive CSS."""
    response = api_client.get("/help")
    assert response.status_code == 200
    content = response.text
    
    # Should have viewport meta tag
    assert 'name="viewport"' in content
    # Should have media queries or responsive framework
    assert "@media" in content or "grid" in content or "flex" in content
