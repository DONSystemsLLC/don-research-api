"""
DON Research API Client Library

A production-quality Python client for interacting with the DON Research API,
designed for academic research institutions using bearer token authentication.

Features:
- Type-safe requests using Pydantic models
- Automatic rate limiting and retry logic
- Both synchronous and asynchronous job support
- Comprehensive error handling
- Progress tracking for long-running jobs

Example usage:
    client = DonResearchClient(
        token="your_institutional_token",
        base_url="https://don-research-api.onrender.com"
    )
    
    # Train QAC model
    job = client.qac.fit(embedding=vectors, params=QACParams(k_nn=8))
    result = client.qac.poll_until_complete(job.id)
    
    # Apply model to new data
    output = client.qac.apply(model_id=result.model_id, embedding=new_vectors)

Author: DON Systems LLC
License: See LICENSE file
"""

import os
import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import logging

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    raise ImportError(
        "The 'requests' library is required. Install with: pip install requests"
    )

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    raise ImportError(
        "The 'pydantic' library is required. Install with: pip install pydantic"
    )


# ============================================================================
# Pydantic Models (matching API schemas)
# ============================================================================

class QACParams(BaseModel):
    """Parameters for QAC model training and application."""
    k_nn: int = Field(default=15, ge=1, le=128, description="Number of nearest neighbors")
    weight: str = Field(default="binary", description="Weight type: 'binary' or 'gaussian'")
    sigma: Optional[float] = Field(default=None, description="Gaussian sigma (if weight='gaussian')")
    reinforce_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Reinforcement rate")
    layers: int = Field(default=50, ge=1, le=100000, description="Number of layers")
    beta: float = Field(default=0.7, ge=0.0, le=10.0, description="Beta parameter for fallback")
    lambda_entropy: float = Field(default=0.05, ge=0.0, le=10.0, description="Lambda entropy for fallback")
    engine: str = Field(default="real_qac", description="Engine type: 'real_qac' or 'laplace'")


class QACJob(BaseModel):
    """Represents a QAC job (fit or apply operation)."""
    id: str
    type: str  # "fit" or "apply"
    status: str  # "queued", "running", "succeeded", "failed"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    model_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class QACModelMeta(BaseModel):
    """Metadata for a trained QAC model."""
    model_id: str
    n_cells: int
    k_nn: int
    weight: str
    reinforce_rate: float
    layers: int
    beta: float
    lambda_entropy: float
    created_at: str
    version: str
    engine: str


class UsageInfo(BaseModel):
    """API usage statistics for the current institution."""
    institution: str
    requests_made: int
    limit: int
    remaining: int
    reset_time: str


# ============================================================================
# Exception Classes
# ============================================================================

class DonResearchAPIError(Exception):
    """Base exception for DON Research API errors."""
    pass


class AuthenticationError(DonResearchAPIError):
    """Raised when authentication fails (invalid token)."""
    pass


class RateLimitError(DonResearchAPIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: int, message: str = None):
        self.retry_after = retry_after
        super().__init__(message or f"Rate limit exceeded. Retry after {retry_after} seconds.")


class JobTimeoutError(DonResearchAPIError):
    """Raised when a job polling timeout is reached."""
    pass


class JobFailedError(DonResearchAPIError):
    """Raised when a job fails on the server."""
    pass


# ============================================================================
# QAC Client
# ============================================================================

class QACClient:
    """Client for QAC (Quantum Adjacency Code) operations."""
    
    def __init__(self, parent: 'DonResearchClient'):
        self.parent = parent
        self._logger = logging.getLogger(f"{__name__}.QACClient")
    
    def fit(
        self,
        embedding: List[List[float]],
        params: Optional[QACParams] = None,
        seed: Optional[int] = None,
        sync: bool = False
    ) -> Union[QACJob, Dict[str, Any]]:
        """
        Train a QAC model on embedding data.
        
        Args:
            embedding: 2D array of shape (n_cells, k_features) with cell embeddings
            params: QAC parameters (uses defaults if None)
            seed: Random seed for reproducibility
            sync: If True, wait for completion and return result immediately
        
        Returns:
            QACJob object with job details if sync=False
            Direct result dict if sync=True
        
        Raises:
            AuthenticationError: If token is invalid
            RateLimitError: If rate limit exceeded
            DonResearchAPIError: For other API errors
        """
        payload = {
            "embedding": embedding,
            "sync": sync
        }
        if params:
            payload["params"] = params.dict()
        if seed is not None:
            payload["seed"] = seed
        
        self._logger.info(f"Submitting QAC fit request (sync={sync}, n_cells={len(embedding)})")
        response = self.parent._request("POST", "/api/v1/quantum/qac/fit", json=payload)
        
        if sync:
            return response
        else:
            return QACJob(**response)
    
    def apply(
        self,
        model_id: str,
        embedding: List[List[float]],
        seed: Optional[int] = None,
        sync: bool = False
    ) -> Union[QACJob, Dict[str, Any]]:
        """
        Apply a trained QAC model to new embedding data.
        
        Args:
            model_id: ID of the trained QAC model
            embedding: 2D array of shape (n_cells, k_features) with cell embeddings
            seed: Random seed for reproducibility
            sync: If True, wait for completion and return result immediately
        
        Returns:
            QACJob object with job details if sync=False
            Direct result dict if sync=True
        """
        payload = {
            "model_id": model_id,
            "embedding": embedding,
            "sync": sync
        }
        if seed is not None:
            payload["seed"] = seed
        
        self._logger.info(f"Submitting QAC apply request (model={model_id}, sync={sync})")
        response = self.parent._request("POST", "/api/v1/quantum/qac/apply", json=payload)
        
        if sync:
            return response
        else:
            return QACJob(**response)
    
    def get_job(self, job_id: str) -> QACJob:
        """
        Get the current status of a QAC job.
        
        Args:
            job_id: ID of the job to query
        
        Returns:
            QACJob object with current status
        """
        response = self.parent._request("GET", f"/api/v1/quantum/qac/jobs/{job_id}")
        return QACJob(**response)
    
    def poll_until_complete(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0
    ) -> Dict[str, Any]:
        """
        Poll a job until it completes or fails.
        
        Args:
            job_id: ID of the job to poll
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait in seconds
        
        Returns:
            Result dictionary from completed job
        
        Raises:
            JobTimeoutError: If timeout is reached
            JobFailedError: If job fails on server
        """
        start_time = time.time()
        self._logger.info(f"Polling job {job_id} (interval={poll_interval}s, timeout={timeout}s)")
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise JobTimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )
            
            job = self.get_job(job_id)
            self._logger.debug(f"Job {job_id}: status={job.status}, progress={job.progress:.1%}")
            
            if job.status == "succeeded":
                self._logger.info(f"Job {job_id} completed successfully")
                return job.result
            
            elif job.status == "failed":
                error_msg = job.error or "Unknown error"
                raise JobFailedError(f"Job {job_id} failed: {error_msg}")
            
            elif job.status in ["queued", "running"]:
                time.sleep(poll_interval)
            
            else:
                raise DonResearchAPIError(
                    f"Job {job_id} has unknown status: {job.status}"
                )
    
    def get_model(self, model_id: str) -> QACModelMeta:
        """
        Get metadata for a trained QAC model.
        
        Args:
            model_id: ID of the model to query
        
        Returns:
            QACModelMeta object with model metadata
        """
        response = self.parent._request("GET", f"/api/v1/quantum/qac/models/{model_id}")
        return QACModelMeta(**response)


# ============================================================================
# Genomics Client
# ============================================================================

class GenomicsClient:
    """Client for genomics operations (DON-GPU compression)."""
    
    def __init__(self, parent: 'DonResearchClient'):
        self.parent = parent
        self._logger = logging.getLogger(f"{__name__}.GenomicsClient")
    
    def compress(
        self,
        gene_names: List[str],
        expression_matrix: List[List[float]],
        cell_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compress gene expression data using DON-GPU fractal clustering.
        
        Args:
            gene_names: List of gene names
            expression_matrix: 2D array (n_cells, n_genes) of expression values
            cell_metadata: Optional metadata dict
        
        Returns:
            Dict with compressed vectors and compression statistics
        """
        payload = {
            "gene_names": gene_names,
            "expression_matrix": expression_matrix
        }
        if cell_metadata:
            payload["cell_metadata"] = cell_metadata
        
        self._logger.info(
            f"Submitting genomics compression (n_cells={len(expression_matrix)}, "
            f"n_genes={len(gene_names)})"
        )
        return self.parent._request("POST", "/api/v1/genomics/compress", json=payload)


# ============================================================================
# Main Client
# ============================================================================

class DonResearchClient:
    """
    Main client for DON Research API.
    
    Handles authentication, rate limiting, retries, and provides access
    to QAC and genomics operations.
    
    Args:
        token: Bearer token for authentication (from institution)
        base_url: API base URL (default: production endpoint)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts for failed requests
        verbose: Enable detailed logging
    
    Example:
        >>> client = DonResearchClient(token=os.getenv("TAMU_API_TOKEN"))
        >>> vectors = [[0.1, 0.2], [0.3, 0.4]]
        >>> job = client.qac.fit(embedding=vectors, sync=False)
        >>> result = client.qac.poll_until_complete(job.id)
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://don-research-api.onrender.com",
        timeout: float = 30.0,
        max_retries: int = 3,
        verbose: bool = False
    ):
        # Get token from parameter or environment
        self.token = token or os.getenv("DON_API_TOKEN") or os.getenv("TAMU_API_TOKEN")
        if not self.token:
            raise ValueError(
                "No API token provided. Set 'token' parameter or "
                "DON_API_TOKEN/TAMU_API_TOKEN environment variable."
            )
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self._logger = logging.getLogger(__name__)
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "DON-Research-Client/1.0"
        })
        
        # Rate limiting tracking
        self._rate_limit: Optional[int] = None
        self._rate_remaining: Optional[int] = None
        self._rate_reset: Optional[str] = None
        
        # Initialize sub-clients
        self.qac = QACClient(self)
        self.genomics = GenomicsClient(self)
        
        self._logger.info(f"DonResearchClient initialized (base_url={base_url})")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Internal method for making HTTP requests with error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
        
        Returns:
            Parsed JSON response
        
        Raises:
            AuthenticationError: If token is invalid (401)
            RateLimitError: If rate limit exceeded (429)
            DonResearchAPIError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        self._logger.debug(f"{method} {url}")
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Update rate limit info from headers
            self._rate_limit = response.headers.get('X-RateLimit-Limit')
            self._rate_remaining = response.headers.get('X-RateLimit-Remaining')
            self._rate_reset = response.headers.get('X-RateLimit-Reset')
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise RateLimitError(retry_after)
            
            # Handle authentication errors
            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid or expired API token. Please check your credentials."
                )
            
            # Handle not found
            if response.status_code == 404:
                raise DonResearchAPIError(f"Resource not found: {endpoint}")
            
            # Handle other errors
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', f"HTTP {response.status_code}")
                except Exception:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                raise DonResearchAPIError(error_msg)
            
            return response.json()
        
        except requests.exceptions.Timeout:
            raise DonResearchAPIError(f"Request timed out after {self.timeout} seconds")
        
        except requests.exceptions.ConnectionError as e:
            raise DonResearchAPIError(f"Connection error: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            raise DonResearchAPIError(f"Request failed: {str(e)}")
    
    def health(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status dict with system information
        """
        return self._request("GET", "/api/v1/health")
    
    def usage(self) -> UsageInfo:
        """
        Get current usage statistics for your institution.
        
        Returns:
            UsageInfo object with rate limit details
        """
        response = self._request("GET", "/api/v1/usage")
        return UsageInfo(**response)
    
    @property
    def rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status from last request.
        
        Returns:
            Dict with limit, remaining, and reset time
        """
        return {
            "limit": self._rate_limit,
            "remaining": self._rate_remaining,
            "reset_time": self._rate_reset
        }
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
