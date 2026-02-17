"""
Logging Configuration Module
Centralized logging setup and configuration
"""

import logging
import logging.handlers
import os
from typing import Optional, Dict, Any


class LoggerSetup:
    """
    Configures and manages application logging
    """
    
    @staticmethod
    def setup_logger(name: str, log_level: str = 'INFO',
                    log_file: Optional[str] = None,
                    format_string: Optional[str] = None) -> logging.Logger:
        """
        Setup logger with console and optional file handlers
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file to log to
            format_string: Custom log format
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Use default format if not provided
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
            )
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', 
                       exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def setup_app_logging(app_name: str, log_dir: str = 'logs',
                         log_level: str = 'INFO',
                         console_output: bool = True) -> Dict[str, logging.Logger]:
        """
        Setup complete application logging
        
        Args:
            app_name: Application name
            log_dir: Directory for log files
            log_level: Default logging level
            console_output: Whether to output to console
            
        Returns:
            Dictionary of configured loggers
        """
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        loggers = {}
        
        # Setup main application logger
        app_log_file = os.path.join(log_dir, f'{app_name}.log')
        loggers['app'] = LoggerSetup.setup_logger(
            app_name,
            log_level=log_level,
            log_file=app_log_file
        )
        
        # Setup error logger
        error_log_file = os.path.join(log_dir, f'{app_name}_errors.log')
        loggers['errors'] = LoggerSetup.setup_logger(
            f'{app_name}_errors',
            log_level='ERROR',
            log_file=error_log_file
        )
        
        # Setup request logger
        request_log_file = os.path.join(log_dir, f'{app_name}_requests.log')
        loggers['requests'] = LoggerSetup.setup_logger(
            f'{app_name}_requests',
            log_level='INFO',
            log_file=request_log_file,
            format_string='%(asctime)s - %(message)s'
        )
        
        # Setup performance logger
        perf_log_file = os.path.join(log_dir, f'{app_name}_performance.log')
        loggers['performance'] = LoggerSetup.setup_logger(
            f'{app_name}_performance',
            log_level='INFO',
            log_file=perf_log_file,
            format_string='%(asctime)s - %(message)s'
        )
        
        return loggers


class ContextLogger:
    """
    Logger wrapper that adds context information
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        """
        Initialize ContextLogger
        
        Args:
            logger: Base logger instance
            context: Context dictionary
        """
        self.logger = logger
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context"""
        if self.context:
            context_str = ' | '.join(f"{k}={v}" for k, v in self.context.items())
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **context):
        """Log debug message"""
        self.context.update(context)
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str, **context):
        """Log info message"""
        self.context.update(context)
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str, **context):
        """Log warning message"""
        self.context.update(context)
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str, **context):
        """Log error message"""
        self.context.update(context)
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str, **context):
        """Log critical message"""
        self.context.update(context)
        self.logger.critical(self._format_message(message))


__all__ = ['LoggerSetup', 'ContextLogger']
