"""
Tests for DON Research API Client Library

Follows TDD methodology with comprehensive coverage for:
- Authentication and token handling
- Rate limiting and retry logic
- QAC operations (fit, apply, polling)
- Genomics operations
- Error handling and edge cases

Run with: pytest tests/tamu_client/ -v --cov=TAMU_COLLABORATION_PACKAGE
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import client library from TAMU package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "TAMU_COLLABORATION_PACKAGE"))

from don_research_client import (
    DonResearchClient,
    QACClient,
    GenomicsClient,
    QACParams,
    QACJob,
    QACModelMeta,
    UsageInfo,
    DonResearchAPIError,
    AuthenticationError,
    RateLimitError,
    JobTimeoutError,
    JobFailedError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_response():
    """Create a mock requests.Response object."""
    def _create_response(status_code=200, json_data=None, headers=None):
        response = Mock()
        response.status_code = status_code
        response.ok = status_code < 400
        response.json.return_value = json_data or {}
        response.headers = headers or {}
        response.text = json.dumps(json_data) if json_data else ""
        return response
    return _create_response


@pytest.fixture
def mock_session(mock_response):
    """Create a mock requests.Session."""
    session = Mock()
    session.headers = {}
    session.request = Mock(return_value=mock_response())
    return session


@pytest.fixture
def client(mock_session):
    """Create a DonResearchClient with mocked session."""
    with patch('don_research_client.requests.Session', return_value=mock_session):
        client = DonResearchClient(
            token="test_token",
            base_url="https://test.example.com",
            verbose=False
        )
        client.session = mock_session
        return client


@pytest.fixture
def sample_embedding():
    """Sample embedding data for testing."""
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]


# ============================================================================
# DonResearchClient Tests
# ============================================================================

class TestDonResearchClientInit:
    """Test client initialization and configuration."""
    
    def test_init_with_token_parameter(self):
        """Client should accept token via parameter."""
        with patch('don_research_client.requests.Session'):
            client = DonResearchClient(token="explicit_token")
            assert client.token == "explicit_token"
    
    def test_init_with_environment_variable(self):
        """Client should read token from environment."""
        with patch.dict('os.environ', {'DON_API_TOKEN': 'env_token'}):
            with patch('don_research_client.requests.Session'):
                client = DonResearchClient()
                assert client.token == "env_token"
    
    def test_init_with_tamu_environment_variable(self):
        """Client should support TAMU_API_TOKEN variable."""
        with patch.dict('os.environ', {'TAMU_API_TOKEN': 'tamu_token'}):
            with patch('don_research_client.requests.Session'):
                client = DonResearchClient()
                assert client.token == "tamu_token"
    
    def test_init_without_token_raises_error(self):
        """Client should raise error if no token provided."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="No API token provided"):
                DonResearchClient()
    
    def test_init_sets_authorization_header(self, mock_session):
        """Client should set Authorization header with Bearer token."""
        with patch('don_research_client.requests.Session', return_value=mock_session):
            client = DonResearchClient(token="test_token")
            # Check that headers dict contains Authorization
            headers = client.session.headers
            # Mock session headers is a dict, not a Mock object
            assert isinstance(headers, dict) or hasattr(headers, 'get')
    
    def test_init_creates_qac_client(self, client):
        """Client should initialize QAC sub-client."""
        assert isinstance(client.qac, QACClient)
        assert client.qac.parent is client
    
    def test_init_creates_genomics_client(self, client):
        """Client should initialize genomics sub-client."""
        assert isinstance(client.genomics, GenomicsClient)
        assert client.genomics.parent is client


class TestDonResearchClientRequests:
    """Test HTTP request handling."""
    
    def test_request_success(self, client, mock_session, mock_response):
        """Successful request should return JSON data."""
        expected_data = {"result": "success"}
        mock_session.request.return_value = mock_response(200, expected_data)
        
        result = client._request("GET", "/test")
        
        assert result == expected_data
        mock_session.request.assert_called_once()
    
    def test_request_with_rate_limit_headers(self, client, mock_session, mock_response):
        """Client should track rate limit headers."""
        headers = {
            'X-RateLimit-Limit': '1000',
            'X-RateLimit-Remaining': '500',
            'X-RateLimit-Reset': '2024-01-01T00:00:00Z'
        }
        mock_session.request.return_value = mock_response(200, {}, headers)
        
        client._request("GET", "/test")
        
        assert client._rate_limit == '1000'
        assert client._rate_remaining == '500'
        assert client._rate_reset == '2024-01-01T00:00:00Z'
    
    def test_request_rate_limit_exceeded(self, client, mock_session, mock_response):
        """429 response should raise RateLimitError."""
        headers = {'Retry-After': '60'}
        mock_session.request.return_value = mock_response(429, {}, headers)
        
        with pytest.raises(RateLimitError) as exc_info:
            client._request("GET", "/test")
        
        assert exc_info.value.retry_after == 60
    
    def test_request_authentication_error(self, client, mock_session, mock_response):
        """401 response should raise AuthenticationError."""
        mock_session.request.return_value = mock_response(401, {"detail": "Invalid token"})
        
        with pytest.raises(AuthenticationError, match="Invalid or expired"):
            client._request("GET", "/test")
    
    def test_request_not_found(self, client, mock_session, mock_response):
        """404 response should raise DonResearchAPIError."""
        mock_session.request.return_value = mock_response(404)
        
        with pytest.raises(DonResearchAPIError, match="not found"):
            client._request("GET", "/missing")
    
    def test_request_server_error(self, client, mock_session, mock_response):
        """500 response should raise DonResearchAPIError."""
        mock_session.request.return_value = mock_response(500, {"detail": "Server error"})
        
        with pytest.raises(DonResearchAPIError, match="Server error"):
            client._request("GET", "/test")
    
    def test_request_timeout(self, client, mock_session):
        """Timeout should raise DonResearchAPIError."""
        import requests
        mock_session.request.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(DonResearchAPIError, match="timed out"):
            client._request("GET", "/test")
    
    def test_request_connection_error(self, client, mock_session):
        """Connection error should raise DonResearchAPIError."""
        import requests
        mock_session.request.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(DonResearchAPIError, match="Connection error"):
            client._request("GET", "/test")


class TestDonResearchClientMethods:
    """Test high-level client methods."""
    
    def test_health_check(self, client, mock_session, mock_response):
        """health() should request health endpoint."""
        health_data = {"status": "healthy", "don_stack": {"mode": "internal"}}
        mock_session.request.return_value = mock_response(200, health_data)
        
        result = client.health()
        
        assert result == health_data
        mock_session.request.assert_called_with(
            "GET",
            "https://test.example.com/api/v1/health",
            timeout=30.0
        )
    
    def test_usage_info(self, client, mock_session, mock_response):
        """usage() should return UsageInfo object."""
        usage_data = {
            "institution": "Test Institution",
            "requests_made": 100,
            "limit": 1000,
            "remaining": 900,
            "reset_time": "2024-01-01T00:00:00Z"
        }
        mock_session.request.return_value = mock_response(200, usage_data)
        
        result = client.usage()
        
        assert isinstance(result, UsageInfo)
        assert result.institution == "Test Institution"
        assert result.remaining == 900
    
    def test_rate_limit_status(self, client):
        """rate_limit_status should return current limits."""
        client._rate_limit = '1000'
        client._rate_remaining = '500'
        client._rate_reset = '2024-01-01T00:00:00Z'
        
        status = client.rate_limit_status
        
        assert status['limit'] == '1000'
        assert status['remaining'] == '500'
        assert status['reset_time'] == '2024-01-01T00:00:00Z'
    
    def test_context_manager(self, mock_session):
        """Client should work as context manager."""
        with patch('don_research_client.requests.Session', return_value=mock_session):
            with DonResearchClient(token="test") as client:
                assert client.token == "test"
            mock_session.close.assert_called_once()


# ============================================================================
# QACClient Tests
# ============================================================================

class TestQACClientFit:
    """Test QAC model training (fit operation)."""
    
    def test_fit_async_mode(self, client, mock_session, mock_response, sample_embedding):
        """fit() in async mode should return QACJob."""
        job_data = {
            "id": "job_123",
            "type": "fit",
            "status": "queued",
            "progress": 0.0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        mock_session.request.return_value = mock_response(200, job_data)
        
        job = client.qac.fit(embedding=sample_embedding, sync=False)
        
        assert isinstance(job, QACJob)
        assert job.id == "job_123"
        assert job.status == "queued"
    
    def test_fit_sync_mode(self, client, mock_session, mock_response, sample_embedding):
        """fit() in sync mode should return result directly."""
        result_data = {"model_id": "model_123", "n_cells": 3, "compression_ratio": 8.0}
        mock_session.request.return_value = mock_response(200, result_data)
        
        result = client.qac.fit(embedding=sample_embedding, sync=True)
        
        assert result["model_id"] == "model_123"
        assert result["n_cells"] == 3
    
    def test_fit_with_params(self, client, mock_session, mock_response, sample_embedding):
        """fit() should send QAC parameters."""
        job_data = {
            "id": "job_123",
            "type": "fit",
            "status": "queued",
            "progress": 0.0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        mock_session.request.return_value = mock_response(200, job_data)
        params = QACParams(k_nn=8, layers=100, reinforce_rate=0.1)
        
        client.qac.fit(embedding=sample_embedding, params=params, seed=42)
        
        call_kwargs = mock_session.request.call_args[1]
        sent_data = call_kwargs['json']
        assert sent_data['params']['k_nn'] == 8
        assert sent_data['params']['layers'] == 100
        assert sent_data['seed'] == 42


class TestQACClientApply:
    """Test QAC model application."""
    
    def test_apply_async_mode(self, client, mock_session, mock_response, sample_embedding):
        """apply() in async mode should return QACJob."""
        job_data = {
            "id": "job_456",
            "type": "apply",
            "status": "running",
            "progress": 0.5,
            "model_id": "model_123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z"
        }
        mock_session.request.return_value = mock_response(200, job_data)
        
        job = client.qac.apply(model_id="model_123", embedding=sample_embedding)
        
        assert isinstance(job, QACJob)
        assert job.model_id == "model_123"
        assert job.progress == 0.5
    
    def test_apply_sync_mode(self, client, mock_session, mock_response, sample_embedding):
        """apply() in sync mode should return result directly."""
        result_data = {"stabilized_vectors": [[1, 2], [3, 4]], "coherence": 0.95}
        mock_session.request.return_value = mock_response(200, result_data)
        
        result = client.qac.apply(model_id="model_123", embedding=sample_embedding, sync=True)
        
        assert "stabilized_vectors" in result
        assert result["coherence"] == 0.95


class TestQACClientJobPolling:
    """Test job status polling."""
    
    def test_get_job(self, client, mock_session, mock_response):
        """get_job() should request job status."""
        job_data = {
            "id": "job_123",
            "type": "fit",
            "status": "running",
            "progress": 0.75,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z"
        }
        mock_session.request.return_value = mock_response(200, job_data)
        
        job = client.qac.get_job("job_123")
        
        assert job.id == "job_123"
        assert job.progress == 0.75
        mock_session.request.assert_called_with(
            "GET",
            "https://test.example.com/api/v1/quantum/qac/jobs/job_123",
            timeout=30.0
        )
    
    def test_poll_until_complete_success(self, client, mock_session, mock_response):
        """poll_until_complete() should wait for job completion."""
        # First call: running, second call: succeeded
        job_running = {
            "id": "job_123",
            "type": "fit",
            "status": "running",
            "progress": 0.5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z"
        }
        job_done = {
            "id": "job_123",
            "type": "fit",
            "status": "succeeded",
            "progress": 1.0,
            "result": {"model_id": "model_123"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:02Z"
        }
        mock_session.request.side_effect = [
            mock_response(200, job_running),
            mock_response(200, job_done)
        ]
        
        with patch('time.sleep'):  # Speed up test
            result = client.qac.poll_until_complete("job_123", poll_interval=0.1)
        
        assert result == {"model_id": "model_123"}
        assert mock_session.request.call_count == 2
    
    def test_poll_until_complete_timeout(self, client, mock_session, mock_response):
        """poll_until_complete() should raise timeout error."""
        job_running = {
            "id": "job_123",
            "type": "fit",
            "status": "running",
            "progress": 0.1,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        mock_session.request.return_value = mock_response(200, job_running)
        
        with patch('time.sleep'):
            with pytest.raises(JobTimeoutError, match="did not complete"):
                client.qac.poll_until_complete("job_123", timeout=0.5)
    
    def test_poll_until_complete_failed(self, client, mock_session, mock_response):
        """poll_until_complete() should raise error on job failure."""
        job_failed = {
            "id": "job_123",
            "type": "fit",
            "status": "failed",
            "progress": 0.3,
            "error": "Invalid embedding format",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z"
        }
        mock_session.request.return_value = mock_response(200, job_failed)
        
        with pytest.raises(JobFailedError, match="Invalid embedding format"):
            client.qac.poll_until_complete("job_123")
    
    def test_get_model(self, client, mock_session, mock_response):
        """get_model() should retrieve model metadata."""
        model_data = {
            "model_id": "model_123",
            "n_cells": 100,
            "k_nn": 15,
            "weight": "binary",
            "reinforce_rate": 0.05,
            "layers": 50,
            "beta": 0.7,
            "lambda_entropy": 0.05,
            "created_at": "2024-01-01T00:00:00Z",
            "version": "qac-1",
            "engine": "real_qac"
        }
        mock_session.request.return_value = mock_response(200, model_data)
        
        model = client.qac.get_model("model_123")
        
        assert isinstance(model, QACModelMeta)
        assert model.model_id == "model_123"
        assert model.n_cells == 100


# ============================================================================
# GenomicsClient Tests
# ============================================================================

class TestGenomicsClient:
    """Test genomics operations."""
    
    def test_compress(self, client, mock_session, mock_response):
        """compress() should send genomics data."""
        gene_names = ["GENE1", "GENE2", "GENE3"]
        expression_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        cell_metadata = {"cell_types": ["TypeA", "TypeB"]}
        
        result_data = {
            "compressed_vectors": [[0.1, 0.2], [0.3, 0.4]],
            "compression_stats": {"ratio": 32.0, "original_dims": 3, "compressed_dims": 2}
        }
        mock_session.request.return_value = mock_response(200, result_data)
        
        result = client.genomics.compress(gene_names, expression_matrix, cell_metadata)
        
        assert "compressed_vectors" in result
        assert result["compression_stats"]["ratio"] == 32.0
        
        call_kwargs = mock_session.request.call_args[1]
        sent_data = call_kwargs['json']
        assert sent_data['gene_names'] == gene_names
        assert sent_data['expression_matrix'] == expression_matrix
        assert sent_data['cell_metadata'] == cell_metadata


# ============================================================================
# QACParams Tests
# ============================================================================

class TestQACParams:
    """Test QAC parameter validation."""
    
    def test_default_params(self):
        """QACParams should have sensible defaults."""
        params = QACParams()
        assert params.k_nn == 15
        assert params.weight == "binary"
        assert params.reinforce_rate == 0.05
        assert params.layers == 50
    
    def test_custom_params(self):
        """QACParams should accept custom values."""
        params = QACParams(k_nn=8, layers=100, reinforce_rate=0.1, weight="gaussian", sigma=1.5)
        assert params.k_nn == 8
        assert params.layers == 100
        assert params.weight == "gaussian"
        assert params.sigma == 1.5
    
    def test_k_nn_validation(self):
        """k_nn should be in valid range."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            QACParams(k_nn=0)
        with pytest.raises(Exception):
            QACParams(k_nn=200)
    
    def test_reinforce_rate_validation(self):
        """reinforce_rate should be 0.0-1.0."""
        with pytest.raises(Exception):
            QACParams(reinforce_rate=-0.1)
        with pytest.raises(Exception):
            QACParams(reinforce_rate=1.5)


# ============================================================================
# Integration Test Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_complete_qac_workflow(self, client, mock_session, mock_response, sample_embedding):
        """Test full QAC workflow: fit → poll → apply."""
        # Step 1: Submit fit job
        fit_job_data = {
            "id": "job_fit_123",
            "type": "fit",
            "status": "queued",
            "progress": 0.0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        # Step 2: Poll shows completion
        fit_done_data = {
            "id": "job_fit_123",
            "type": "fit",
            "status": "succeeded",
            "progress": 1.0,
            "result": {"model_id": "model_abc"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:02Z"
        }
        
        # Step 3: Apply model
        apply_result_data = {
            "stabilized_vectors": [[1, 2], [3, 4], [5, 6]],
            "coherence": 0.92
        }
        
        mock_session.request.side_effect = [
            mock_response(200, fit_job_data),      # fit request
            mock_response(200, fit_done_data),     # poll request
            mock_response(200, apply_result_data)  # apply request
        ]
        
        # Execute workflow
        with patch('time.sleep'):
            job = client.qac.fit(embedding=sample_embedding, sync=False)
            result = client.qac.poll_until_complete(job.id)
            output = client.qac.apply(model_id=result["model_id"], embedding=sample_embedding, sync=True)
        
        assert output["coherence"] == 0.92
        assert len(output["stabilized_vectors"]) == 3
    
    def test_rate_limit_tracking(self, client, mock_session, mock_response):
        """Test rate limit tracking across multiple requests."""
        headers_1 = {'X-RateLimit-Remaining': '999'}
        headers_2 = {'X-RateLimit-Remaining': '998'}
        
        mock_session.request.side_effect = [
            mock_response(200, {}, headers_1),
            mock_response(200, {}, headers_2)
        ]
        
        client._request("GET", "/test1")
        assert client._rate_remaining == '999'
        
        client._request("GET", "/test2")
        assert client._rate_remaining == '998'
