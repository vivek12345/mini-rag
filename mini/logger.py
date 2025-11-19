"""
Logging module for mini-rag.
Provides a centralized logger that can be used across all modules.
"""

import logging
import os
import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Log level colors
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[91m'   # Bright Red
    
    # Component colors
    TIMESTAMP = '\033[90m'  # Dark Gray
    NAME = '\033[94m'       # Light Blue


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log output."""
    
    def __init__(self, use_colors: bool = True, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize the colored formatter.
        
        Args:
            use_colors: Whether to use colors. Defaults to True if output is a TTY.
            fmt: Format string for the log message.
            datefmt: Date format string.
        """
        fmt = fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        datefmt = datefmt or '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get base formatted message
        log_message = super().format(record)
        
        if not self.use_colors:
            return log_message
        
        # Add colors based on log level
        level_colors = {
            'DEBUG': Colors.DEBUG,
            'INFO': Colors.INFO,
            'WARNING': Colors.WARNING,
            'ERROR': Colors.ERROR,
            'CRITICAL': Colors.CRITICAL + Colors.BOLD,
        }
        
        level_color = level_colors.get(record.levelname, Colors.RESET)
        
        # Colorize different parts of the log message
        parts = log_message.split(' - ', 3)
        if len(parts) == 4:
            timestamp, name, levelname, message = parts
            colored_timestamp = f"{Colors.TIMESTAMP}{timestamp}{Colors.RESET}"
            colored_name = f"{Colors.NAME}{name}{Colors.RESET}"
            colored_level = f"{level_color}{levelname}{Colors.RESET}"
            return f"{colored_timestamp} - {colored_name} - {colored_level} - {message}"
        else:
            # Fallback: just colorize the level name
            return log_message.replace(
                record.levelname,
                f"{level_color}{record.levelname}{Colors.RESET}"
            )


class Logger:
    """
    Centralized logger for mini-rag package.
    
    Examples:
        from mini.logger import logger
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    """
    
    _instance: Optional['Logger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Logger._initialized:
            return
        
        # Get log level from environment variable or default to INFO
        log_level = os.getenv("MINI_RAG_LOG_LEVEL", "INFO").upper()
        
        # Create logger
        self.logger = logging.getLogger("mini_rag")
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create console handler
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Create colored formatter
            formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(handler)
        
        Logger._initialized = True
    
    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(message)
    
    def set_level(self, level: str):
        """Set the logging level."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)


# Create singleton instance
logger = Logger()

