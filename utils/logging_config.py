# utils/logging_config.py
import logging
import logging.config
import sys # Needed to set stream handler target

# In a real application, this function would be called early in the startup process
# and would likely accept the loaded configuration object.
def setup_logging(config):
    """
    Sets up the logging configuration based on the provided config object.
    """
    # Use dictConfig for more complex configurations if needed later
    # For now, simple basicConfig based on config settings

    log_level_str = config.logging.get('level', 'INFO').upper()
    log_format = config.logging.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_datefmt = config.logging.get('datefmt', "%Y-%m-%d %H:%M:%S")

    # Map string level from config to logging module constants
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Basic Configuration - logs to console (stderr by default)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_datefmt,
        # You can add stream=sys.stdout if you prefer logging to stdout
        # handlers=[logging.StreamHandler(sys.stdout)] # Example if you need explicit handlers
    )

    # Optional: Add handler for specific requirements (e.g., file output)
    # from logging.handlers import RotatingFileHandler
    # file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024*5, backupCount=5)
    # file_handler.setLevel(log_level)
    # formatter = logging.Formatter(log_format, datefmt=log_datefmt)
    # file_handler.setFormatter(formatter)
    # logging.getLogger().addHandler(file_handler)

    logging.info("Logging configured successfully.")

# Example usage (optional - can be kept for testing this module)
if __name__ == "__main__":
    # For testing this module directly, we need a mock config object
    class MockConfig:
        def __init__(self):
            self.logging = {
                'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }

    mock_config = MockConfig()
    setup_logging(mock_config)

    # Now you can use the logger
    logger = logging.getLogger(__name__) # Get a logger for this module
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")