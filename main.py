"""
SOMEBODY - Smart Omni-functional Machine for Enhanced Brainstorming, Observation, 
and Decision-support for Your Needs

Main application entry point
"""

import os
import logging
from datetime import datetime
from integration.server import start_server
from integration.controller import SomebodyController

# Konfigurasi logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/somebody_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to start the SOMEBODY application"""
    try:
        logger.info("Starting SOMEBODY system...")
        
        # Initialize controller
        controller = SomebodyController()
        controller.initialize()
        controller.load_state()
        
        # Start the integration API server
        start_server()
        
    except KeyboardInterrupt:
        logger.info("SOMEBODY system shutdown initiated by user")
        controller = SomebodyController()
        controller.shutdown()
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}", exc_info=True)
    finally:
        logger.info("SOMEBODY system shutdown complete")

if __name__ == "__main__":
    main()