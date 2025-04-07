"""
Common utility functions for SOMEBODY
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def timeit(func):
    """Decorator to measure the execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

def is_valid_file(file_path):
    """Check if a file exists and is accessible"""
    import os
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def download_file(url, save_path):
    """Download a file from URL"""
    import requests
    from tqdm import tqdm
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download with progress bar
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        return True
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        return False