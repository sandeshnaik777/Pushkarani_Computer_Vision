"""
API Documentation Module
Utilities for API documentation and schema generation
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class APIDocumentation:
    """Generates API documentation"""
    
    @staticmethod
    def generate_endpoint_docs(endpoint_name: str, method: str,
                             description: str, params: Dict[str, Dict],
                             response_schema: Dict) -> Dict[str, Any]:
        """
        Generate documentation for an endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            method: HTTP method
            description: Endpoint description
            params: Parameter definitions
            response_schema: Response schema
            
        Returns:
            Documentation dictionary
        """
        return {
            'endpoint': endpoint_name,
            'method': method,
            'description': description,
            'parameters': params,
            'response_schema': response_schema,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def generate_api_spec(title: str, version: str,
                         description: str, endpoints: List[Dict]) -> Dict[str, Any]:
        """
        Generate complete API specification
        
        Args:
            title: API title
            version: API version
            description: API description
            endpoints: List of endpoint documentation
            
        Returns:
            API specification
        """
        return {
            'title': title,
            'version': version,
            'description': description,
            'endpoints': endpoints,
            'generated_at': datetime.utcnow().isoformat()
        }


class SchemaBuilder:
    """Builds JSON schemas"""
    
    @staticmethod
    def string_schema(description: str = "", min_length: int = None,
                     max_length: int = None, pattern: str = None) -> Dict[str, Any]:
        """Build string schema"""
        schema = {
            'type': 'string',
            'description': description
        }
        if min_length is not None:
            schema['minLength'] = min_length
        if max_length is not None:
            schema['maxLength'] = max_length
        if pattern:
            schema['pattern'] = pattern
        return schema
    
    @staticmethod
    def number_schema(description: str = "", min_value: float = None,
                     max_value: float = None) -> Dict[str, Any]:
        """Build number schema"""
        schema = {
            'type': 'number',
            'description': description
        }
        if min_value is not None:
            schema['minimum'] = min_value
        if max_value is not None:
            schema['maximum'] = max_value
        return schema
    
    @staticmethod
    def object_schema(properties: Dict[str, Dict],
                     required: List[str] = None) -> Dict[str, Any]:
        """Build object schema"""
        schema = {
            'type': 'object',
            'properties': properties
        }
        if required:
            schema['required'] = required
        return schema
    
    @staticmethod
    def array_schema(items_schema: Dict[str, Any],
                    min_items: int = None,
                    max_items: int = None) -> Dict[str, Any]:
        """Build array schema"""
        schema = {
            'type': 'array',
            'items': items_schema
        }
        if min_items is not None:
            schema['minItems'] = min_items
        if max_items is not None:
            schema['maxItems'] = max_items
        return schema


__all__ = ['APIDocumentation', 'SchemaBuilder']
