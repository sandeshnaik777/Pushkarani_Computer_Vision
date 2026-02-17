"""
Performance Monitoring Module
Tracks and reports on application performance metrics
"""

import time
import psutil
import logging
from typing import Dict, List, Any
from collections import deque
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors application performance metrics
    Tracks response times, resource usage, and request patterns
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize PerformanceMonitor
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.response_times = deque(maxlen=max_history)
        self.request_sizes = deque(maxlen=max_history)
        self.response_sizes = deque(maxlen=max_history)
        self.error_log = deque(maxlen=max_history)
        self.lock = Lock()
        
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.total_bytes_in = 0
        self.total_bytes_out = 0
        self.process = psutil.Process()
    
    def record_request(self, endpoint: str, method: str, 
                      request_size: int = 0) -> str:
        """
        Record the start of a request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            request_size: Size of request in bytes
            
        Returns:
            Request ID
        """
        with self.lock:
            self.total_requests += 1
            self.total_bytes_in += request_size
            self.request_sizes.append(request_size)
            
            request_id = f"{endpoint}_{self.total_requests}_{int(time.time() * 1000)}"
            return request_id
    
    def record_response(self, request_id: str, duration: float, 
                       response_size: int = 0, status_code: int = 200,
                       error: str = None):
        """
        Record the completion of a request
        
        Args:
            request_id: Request ID from record_request
            duration: Time taken to process (seconds)
            response_size: Size of response in bytes
            status_code: HTTP status code
            error: Error message if any
        """
        with self.lock:
            self.response_times.append(duration)
            self.total_bytes_out += response_size
            self.response_sizes.append(response_size)
            
            if status_code >= 400:
                self.total_errors += 1
                if error:
                    self.error_log.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'request_id': request_id,
                        'status_code': status_code,
                        'error': error
                    })
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage in MB
        
        Returns:
            Dictionary with memory stats
        """
        try:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'percent': self.process.memory_percent()
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """
        Get response time statistics
        
        Returns:
            Dictionary with stats
        """
        import numpy as np
        
        if not self.response_times:
            return {
                'count': 0,
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        times = np.array(list(self.response_times))
        
        return {
            'count': len(times),
            'avg': float(np.mean(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'p50': float(np.percentile(times, 50)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99))
        }
    
    def get_throughput(self) -> Dict[str, float]:
        """
        Get throughput statistics
        
        Returns:
            Dictionary with throughput stats
        """
        uptime = self.get_uptime()
        
        if uptime == 0:
            return {
                'requests_per_second': 0.0,
                'bytes_in_per_second': 0.0,
                'bytes_out_per_second': 0.0,
                'total_bandwidth_mb': 0.0
            }
        
        requests_per_second = self.total_requests / uptime
        bytes_in_per_second = self.total_bytes_in / uptime
        bytes_out_per_second = self.total_bytes_out / uptime
        total_bandwidth = (self.total_bytes_in + self.total_bytes_out) / (1024 * 1024)
        
        return {
            'requests_per_second': requests_per_second,
            'bytes_in_per_second': bytes_in_per_second,
            'bytes_out_per_second': bytes_out_per_second,
            'total_bandwidth_mb': total_bandwidth
        }
    
    def get_error_rate(self) -> float:
        """
        Get error rate percentage
        
        Returns:
            Error rate percentage
        """
        if self.total_requests == 0:
            return 0.0
        
        return (self.total_errors / self.total_requests) * 100
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        with self.lock:
            return list(self.error_log)[-limit:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Returns:
            Dictionary with all key metrics
        """
        memory = self.get_memory_usage()
        response_times = self.get_response_time_stats()
        throughput = self.get_throughput()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': self.get_uptime(),
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate_percent': self.get_error_rate(),
            'cpu_usage_percent': self.get_cpu_usage(),
            'memory': {
                'rss_mb': f"{memory['rss_mb']:.2f}",
                'vms_mb': f"{memory['vms_mb']:.2f}",
                'percent': f"{memory['percent']:.2f}"
            },
            'response_times': {
                'count': response_times['count'],
                'avg_seconds': f"{response_times['avg']:.4f}",
                'min_seconds': f"{response_times['min']:.4f}",
                'max_seconds': f"{response_times['max']:.4f}",
                'p50_seconds': f"{response_times['p50']:.4f}",
                'p95_seconds': f"{response_times['p95']:.4f}",
                'p99_seconds': f"{response_times['p99']:.4f}"
            },
            'throughput': {
                'requests_per_second': f"{throughput['requests_per_second']:.2f}",
                'bytes_in_per_second': f"{throughput['bytes_in_per_second']:.2f}",
                'bytes_out_per_second': f"{throughput['bytes_out_per_second']:.2f}",
                'total_bandwidth_mb': f"{throughput['total_bandwidth_mb']:.2f}"
            },
            'data_transferred': {
                'total_in_mb': f"{self.total_bytes_in / (1024 * 1024):.2f}",
                'total_out_mb': f"{self.total_bytes_out / (1024 * 1024):.2f}"
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.request_times.clear()
            self.response_times.clear()
            self.request_sizes.clear()
            self.response_sizes.clear()
            self.error_log.clear()
            
            self.start_time = time.time()
            self.total_requests = 0
            self.total_errors = 0
            self.total_bytes_in = 0
            self.total_bytes_out = 0
            
            logger.info("Performance metrics reset")


class RequestTimer:
    """Context manager for timing requests"""
    
    def __init__(self, monitor: PerformanceMonitor, endpoint: str, 
                 method: str = 'GET', request_size: int = 0):
        self.monitor = monitor
        self.endpoint = endpoint
        self.method = method
        self.request_size = request_size
        self.request_id = None
        self.start_time = None
        self.response_size = 0
        self.status_code = 200
        self.error = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.request_id = self.monitor.record_request(
            self.endpoint, self.method, self.request_size
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.status_code = 500
            self.error = str(exc_val)
        
        self.monitor.record_response(
            self.request_id, duration, self.response_size,
            self.status_code, self.error
        )
        
        return False
