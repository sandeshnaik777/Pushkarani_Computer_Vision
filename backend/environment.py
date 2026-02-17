"""
Environment and System Utilities Module
Handle environment variables and system information
"""

import os
import sys
import platform
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """
    Manages environment variables with defaults and validation
    """
    
    @staticmethod
    def get_env(key: str, default: Any = None, var_type: type = str) -> Any:
        """
        Get environment variable with type conversion
        
        Args:
            key: Environment variable name
            default: Default value if not found
            var_type: Type to convert value to
            
        Returns:
            Environment variable value or default
        """
        value = os.getenv(key)
        
        if value is None:
            return default
        
        try:
            if var_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif var_type == int:
                return int(value)
            elif var_type == float:
                return float(value)
            else:
                return var_type(value)
        except Exception as e:
            logger.error(f"Error converting env var {key} to {var_type}: {str(e)}")
            return default
    
    @staticmethod
    def require_env(key: str, var_type: type = str) -> Any:
        """
        Get required environment variable
        
        Args:
            key: Environment variable name
            var_type: Type to convert to
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If variable not found
        """
        value = os.getenv(key)
        
        if value is None:
            raise ValueError(f"Required environment variable not found: {key}")
        
        try:
            if var_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif var_type == int:
                return int(value)
            elif var_type == float:
                return float(value)
            else:
                return var_type(value)
        except Exception as e:
            raise ValueError(f"Error converting env var {key} to {var_type}: {str(e)}")
    
    @staticmethod
    def get_all_env_vars() -> Dict[str, str]:
        """
        Get all environment variables
        
        Returns:
            Dictionary of all environment variables
        """
        return dict(os.environ)
    
    @staticmethod
    def set_env(key: str, value: Any):
        """
        Set environment variable
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        os.environ[key] = str(value)
        logger.debug(f"Environment variable set: {key}")
    
    @staticmethod
    def get_env_prefix(prefix: str) -> Dict[str, str]:
        """
        Get all environment variables with given prefix
        
        Args:
            prefix: Prefix to filter by
            
        Returns:
            Dictionary of matching variables
        """
        return {
            k: v for k, v in os.environ.items()
            if k.startswith(prefix)
        }


class SystemInfo:
    """
    Provides system and platform information
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information
        
        Returns:
            Dictionary with system information
        """
        return {
            'platform': platform.system(),
            'platform_version': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'hostname': platform.node(),
            'python_executable': sys.executable,
            'max_int': sys.maxsize
        }
    
    @staticmethod
    def get_python_info() -> Dict[str, Any]:
        """Get Python information"""
        return {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': sys.executable,
            'path': sys.path,
            'prefix': sys.prefix
        }
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get platform information"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'node': platform.node()
        }
    
    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows"""
        return platform.system() == 'Windows'
    
    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux"""
        return platform.system() == 'Linux'
    
    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS"""
        return platform.system() == 'Darwin'
    
    @staticmethod
    def get_working_directory() -> str:
        """Get current working directory"""
        return os.getcwd()
    
    @staticmethod
    def get_home_directory() -> str:
        """Get user home directory"""
        return os.path.expanduser('~')
    
    @staticmethod
    def get_temp_directory() -> str:
        """Get system temporary directory"""
        import tempfile
        return tempfile.gettempdir()


class PathManager:
    """
    Manages file paths and directories
    """
    
    @staticmethod
    def join_paths(*paths: str) -> str:
        """
        Join path components
        
        Args:
            *paths: Path components
            
        Returns:
            Joined path
        """
        return os.path.join(*paths)
    
    @staticmethod
    def get_absolute_path(path: str) -> str:
        """
        Get absolute path
        
        Args:
            path: Relative or absolute path
            
        Returns:
            Absolute path
        """
        return os.path.abspath(path)
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize path for current platform
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized path
        """
        return os.path.normpath(path)
    
    @staticmethod
    def is_absolute_path(path: str) -> bool:
        """
        Check if path is absolute
        
        Args:
            path: Path to check
            
        Returns:
            True if absolute path
        """
        return os.path.isabs(path)
    
    @staticmethod
    def get_relative_path(path: str, start: str = '.') -> str:
        """
        Get relative path
        
        Args:
            path: Target path
            start: Start path (default current dir)
            
        Returns:
            Relative path
        """
        return os.path.relpath(path, start)
    
    @staticmethod
    def expand_user_path(path: str) -> str:
        """
        Expand ~ to user home directory
        
        Args:
            path: Path with ~
            
        Returns:
            Expanded path
        """
        return os.path.expanduser(path)


__all__ = [
    'EnvironmentManager',
    'SystemInfo',
    'PathManager'
]
