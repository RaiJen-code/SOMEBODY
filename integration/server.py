"""
Server runner for SOMEBODY integration API
"""

import logging
import uvicorn
from config.settings import API_HOST, API_PORT, DEBUG

logger = logging.getLogger(__name__)

def start_server():
    """Start the FastAPI server"""
    logger.info(f"Starting SOMEBODY Integration API at {API_HOST}:{API_PORT}")
    uvicorn.run(
        "integration.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()