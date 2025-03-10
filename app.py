#!/usr/bin/env python3
"""
Main entry point for MAB Web Design Testing application.
This imports the actual app from the mab_web_app module.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('startup')

def main():
    """Main entry point for the application"""
    try:
        logger.info("Starting MAB Web Design Testing application")
        
        # Import the Flask app
        from mab_web_app.app import app
        
        # Get port from environment variable or use default
        port = int(os.getenv('PORT', 5000))
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 