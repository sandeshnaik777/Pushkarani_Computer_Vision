"""
Notification and Alerts Module
Handles notifications, alerts, and event notifications
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """Represents a single alert"""
    
    def __init__(self, title: str, message: str, level: AlertLevel,
                 details: Dict[str, Any] = None):
        self.title = title
        self.message = message
        self.level = level
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.acknowledged = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'message': self.message,
            'level': self.level.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }
    
    def acknowledge(self):
        """Mark alert as acknowledged"""
        self.acknowledged = True


class AlertManager:
    """Manages application alerts"""
    
    def __init__(self):
        self.alerts = []
        self.alert_handlers = []
        self.alert_history = []
        self.max_alerts = 1000
    
    def create_alert(self, title: str, message: str,
                    level: AlertLevel, details: Dict[str, Any] = None) -> Alert:
        """
        Create and register an alert
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level
            details: Additional details
            
        Returns:
            Created alert object
        """
        alert = Alert(title, message, level, details)
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Keep history limited
        if len(self.alert_history) > self.max_alerts:
            self.alert_history.pop(0)
        
        # Notify handlers
        self._dispatch_alert(alert)
        
        logger.info(f"Alert created: {title} ({level.value})")
        
        return alert
    
    def _dispatch_alert(self, alert: Alert):
        """Dispatch alert to all registered handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error dispatching alert: {str(e)}")
    
    def register_handler(self, handler: Callable):
        """
        Register alert handler
        
        Args:
            handler: Callable that receives Alert object
        """
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self, level: AlertLevel = None) -> List[Alert]:
        """
        Get active alerts
        
        Args:
            level: Filter by level (optional)
            
        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts if not a.acknowledged]
        
        if level:
            active = [a for a in active if a.level == level]
        
        return active
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by level"""
        summary = {level.value: 0 for level in AlertLevel}
        
        for alert in self.get_active_alerts():
            summary[alert.level.value] += 1
        
        return summary
    
    def clear_alerts(self, level: AlertLevel = None):
        """Clear alerts"""
        if level:
            self.alerts = [a for a in self.alerts if a.level != level]
        else:
            self.alerts.clear()
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return [a.to_dict() for a in self.alert_history[-limit:]]


class Notification:
    """Represents a notification"""
    
    def __init__(self, event_type: str, message: str,
                 recipient: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.message = message
        self.recipient = recipient
        self.data = data or {}
        self.timestamp = datetime.utcnow()
        self.sent = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'message': self.message,
            'recipient': self.recipient,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'sent': self.sent
        }


class NotificationManager:
    """Manages notifications"""
    
    def __init__(self):
        self.notifications = []
        self.notification_handlers = {
            'email': None,
            'sms': None,
            'webhook': None,
            'log': self._log_notification
        }
        self.notification_history = []
    
    def send_notification(self, event_type: str, message: str,
                         recipient: str, methods: List[str] = None,
                         data: Dict[str, Any] = None):
        """
        Send notification via specified methods
        
        Args:
            event_type: Type of event
            message: Notification message
            recipient: Recipient identifier
            methods: Delivery methods ('email', 'sms', 'webhook', 'log')
            data: Additional data
        """
        if methods is None:
            methods = ['log']
        
        notification = Notification(event_type, message, recipient, data)
        self.notifications.append(notification)
        self.notification_history.append(notification)
        
        # Send via requested methods
        for method in methods:
            handler = self.notification_handlers.get(method)
            if handler:
                try:
                    handler(notification)
                    notification.sent = True
                except Exception as e:
                    logger.error(f"Error sending {method} notification: {str(e)}")
        
        return notification
    
    def register_handler(self, method: str, handler: Callable):
        """
        Register notification handler
        
        Args:
            method: Notification method name
            handler: Callable that sends notification
        """
        self.notification_handlers[method] = handler
    
    def _log_notification(self, notification: Notification):
        """Log notification"""
        logger.info(f"Notification: {notification.event_type} - {notification.message}")
    
    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get notification history"""
        return [n.to_dict() for n in self.notification_history[-limit:]]


class EventEmitter:
    """Simple event emitter"""
    
    def __init__(self):
        self.listeners = {}
    
    def on(self, event: str, handler: Callable):
        """Register event handler"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Unregister event handler"""
        if event in self.listeners:
            self.listeners[event].remove(handler)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit event to all handlers"""
        if event in self.listeners:
            for handler in self.listeners[event]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {str(e)}")
    
    def once(self, event: str, handler: Callable):
        """Register one-time event handler"""
        def wrapper(*args, **kwargs):
            handler(*args, **kwargs)
            self.off(event, wrapper)
        
        self.on(event, wrapper)


__all__ = [
    'AlertLevel',
    'Alert',
    'AlertManager',
    'Notification',
    'NotificationManager',
    'EventEmitter'
]
