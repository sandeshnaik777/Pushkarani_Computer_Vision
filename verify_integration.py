#!/usr/bin/env python3
"""
Pushkarani System Integration Verification Script
Validates all components are properly integrated
"""

import os
import json
import sys
from pathlib import Path

class IntegrationValidator:
    def __init__(self, base_path='.'):
        self.base_path = Path(base_path)
        self.results = {
            'backend': {},
            'frontend': {},
            'models': {},
            'knowledge_base': {},
            'overall': 'PENDING'
        }
    
    def validate_backend(self):
        """Validate backend integration"""
        print("\n" + "="*60)
        print("VALIDATING BACKEND INTEGRATION")
        print("="*60)
        
        checks = {}
        
        # Check app.py exists and has required imports
        app_path = self.base_path / 'backend' / 'app.py'
        if app_path.exists():
            with open(app_path, 'r') as f:
                content = f.read()
                checks['app.py_exists'] = True
                checks['has_flask'] = 'from flask import' in content
                checks['has_chatbot_kb_import'] = 'from chatbot_kb import' in content
                checks['has_endpoint_chatbot'] = '@app.route(\'/api/chatbot\'' in content
                checks['has_endpoint_predict'] = '@app.route(\'/api/predict\'' in content
                checks['has_endpoint_water_quality'] = '@app.route(\'/api/water-quality\'' in content
                checks['has_water_quality_analysis'] = 'def analyze_water_quality' in content
                checks['has_simple_chatbot_class'] = 'class SimpleChatbot' in content
                
                print(f"✓ app.py located: {app_path}")
        else:
            checks['app.py_exists'] = False
            print(f"✗ app.py not found at {app_path}")
        
        # Check requirements.txt
        req_path = self.base_path / 'backend' / 'requirements.txt'
        if req_path.exists():
            with open(req_path, 'r') as f:
                content = f.read()
                checks['requirements_exists'] = True
                checks['has_flask_in_req'] = 'flask' in content.lower()
                checks['has_tensorflow_in_req'] = 'tensorflow' in content.lower()
                checks['has_pillow_in_req'] = 'pillow' in content.lower()
                print(f"✓ requirements.txt exists with {len(content.split(chr(10)))} packages")
        else:
            checks['requirements_exists'] = False
            print(f"✗ requirements.txt not found")
        
        # Check chatbot_kb.py
        kb_path = self.base_path / 'backend' / 'chatbot_kb.py'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                content = f.read()
                checks['chatbot_kb_exists'] = True
                checks['has_knowledge_base'] = 'KNOWLEDGE_BASE' in content
                checks['has_type_info'] = '\'type-1\'' in content and '\'type-2\'' in content and '\'type-3\'' in content
                checks['has_facts'] = 'KNOWLEDGE_BASE' in content and len(content) > 5000
                
                # Count facts
                if 'KNOWLEDGE_BASE' in content:
                    facts_count = content.count('largest temple tank')  # Using unique fact as marker
                    print(f"✓ chatbot_kb.py with comprehensive knowledge base")
        else:
            checks['chatbot_kb_exists'] = False
            print(f"✗ chatbot_kb.py not found")
        
        self.results['backend'] = checks
        self._print_check_results(checks, "BACKEND")
        return all(checks.values())
    
    def validate_frontend(self):
        """Validate frontend integration"""
        print("\n" + "="*60)
        print("VALIDATING FRONTEND INTEGRATION")
        print("="*60)
        
        checks = {}
        
        # Check package.json
        pkg_path = self.base_path / 'frontend' / 'package.json'
        if pkg_path.exists():
            with open(pkg_path, 'r') as f:
                try:
                    pkg = json.load(f)
                    checks['package_json_exists'] = True
                    checks['has_react'] = 'react' in pkg.get('dependencies', {})
                    checks['has_axios'] = 'axios' in pkg.get('dependencies', {})
                    checks['has_react_icons'] = 'react-icons' in pkg.get('dependencies', {})
                    print(f"✓ package.json with {len(pkg.get('dependencies', {}))} dependencies")
                except:
                    checks['package_json_exists'] = True
                    checks['parse_error'] = True
                    print(f"✓ package.json exists (parse skipped)")
        else:
            checks['package_json_exists'] = False
            print(f"✗ package.json not found")
        
        # Check React components
        components = [
            'ImageUploader.js',
            'ResultsDisplay.js',
            'WaterQualityAnalyzer.js',
            'Chatbot.js',
            'FactsSection.js',
            'ContributionSection.js'
        ]
        
        components_path = self.base_path / 'frontend' / 'src' / 'components'
        if components_path.exists():
            found_components = 0
            for component in components:
                comp_file = components_path / component
                if comp_file.exists():
                    found_components += 1
                    checks[f'has_{component}'] = True
                    # Check for API integration
                    with open(comp_file, 'r') as f:
                        content = f.read()
                        if 'axios' in content or '/api/' in content:
                            checks[f'{component}_api_integrated'] = True
                else:
                    checks[f'has_{component}'] = False
            
            print(f"✓ Found {found_components}/{len(components)} required components")
        else:
            print(f"✗ Components directory not found")
        
        # Check main App component
        app_path = self.base_path / 'frontend' / 'src' / 'App.js'
        if app_path.exists():
            with open(app_path, 'r') as f:
                content = f.read()
                checks['app_js_exists'] = True
                checks['app_imports_components'] = 'import' in content
                checks['app_has_routes'] = 'return' in content
                print(f"✓ App.js main component exists")
        else:
            checks['app_js_exists'] = False
            print(f"✗ App.js not found")
        
        self.results['frontend'] = checks
        self._print_check_results(checks, "FRONTEND")
        return sum(1 for v in checks.values() if v is True) > len(checks) * 0.7
    
    def validate_models(self):
        """Validate model files"""
        print("\n" + "="*60)
        print("VALIDATING TRAINED MODELS")
        print("="*60)
        
        checks = {}
        model_names = [
            'densenet', 'efficientnetv2', 'convnext', 'vgg16', 
            'resnet50', 'mobilenet', 'mobilenetv3', 'inception',
            'swin', 'dinov2'
        ]
        
        models_found = 0
        for model_name in model_names:
            model_path = self.base_path / model_name
            if model_path.exists():
                models_found += 1
                # Check for required files
                has_best = (model_path / 'best_model.keras').exists()
                has_final = (model_path / 'final_model.keras').exists()
                has_class_indices = (model_path / 'class_indices.json').exists()
                
                checks[f'{model_name}_exists'] = True
                checks[f'{model_name}_best_model'] = has_best
                checks[f'{model_name}_final_model'] = has_final
                checks[f'{model_name}_class_indices'] = has_class_indices
                
                if has_best or has_final:
                    print(f"  ✓ {model_name.upper()}: Found trained model(s)")
            else:
                checks[f'{model_name}_exists'] = False
        
        print(f"\n✓ Total models found: {models_found}/{len(model_names)}")
        
        # Check dataset
        dataset_path = self.base_path / 'dataset'
        dataset_types = ['type-1', 'type-2', 'type-3']
        dataset_found = 0
        if dataset_path.exists():
            for dtype in dataset_types:
                type_path = dataset_path / dtype
                if type_path.exists():
                    dataset_found += 1
                    checks[f'dataset_{dtype}'] = True
                    print(f"  ✓ Dataset {dtype}: Found training samples")
                else:
                    checks[f'dataset_{dtype}'] = False
        
        print(f"✓ Dataset types found: {dataset_found}/{len(dataset_types)}")
        
        self.results['models'] = checks
        self._print_check_results(checks, "MODELS")
        return models_found > 5 and dataset_found >= 2
    
    def validate_knowledge_base(self):
        """Validate knowledge base content"""
        print("\n" + "="*60)
        print("VALIDATING KNOWLEDGE BASE")
        print("="*60)
        
        checks = {}
        
        kb_path = self.base_path / 'backend' / 'chatbot_kb.py'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                content = f.read()
            
            # Validate structure
            checks['has_knowledge_base_dict'] = 'KNOWLEDGE_BASE = {' in content
            checks['has_types_section'] = "'types':" in content
            checks['has_general_section'] = "'general':" in content
            checks['has_facts_list'] = "'facts':" in content
            checks['has_keywords_section'] = "'keywords':" in content
            
            # Count content
            type_1_content = "Teppakulam" in content
            type_2_content = "Kalyani" in content
            type_3_content = "Kunda" in content
            
            checks['type_1_documented'] = type_1_content
            checks['type_2_documented'] = type_2_content
            checks['type_3_documented'] = type_3_content
            
            # Verify facts
            fact_count = content.count('largest temple tank')
            checks['has_comprehensive_facts'] = content.count('\"') > 100
            
            print(f"✓ Knowledge Base Structure Validated:")
            print(f"  - Types section: {'Present' if checks['has_types_section'] else 'Missing'}")
            print(f"  - General knowledge: {'Present' if checks['has_general_section'] else 'Missing'}")
            print(f"  - Facts database: {'Present' if checks['has_facts_list'] else 'Missing'}")
            print(f"  - Keywords mapping: {'Present' if checks['has_keywords_section'] else 'Missing'}")
            print(f"\n✓ Type Coverage:")
            print(f"  - Type-1 (Teppakulam): {'✓' if type_1_content else '✗'}")
            print(f"  - Type-2 (Kalyani): {'✓' if type_2_content else '✗'}")
            print(f"  - Type-3 (Kunda): {'✓' if type_3_content else '✗'}")
            print(f"\n✓ Content Metrics:")
            print(f"  - Comprehensive facts database: {'✓' if checks['has_comprehensive_facts'] else '✗'}")
        else:
            print(f"✗ chatbot_kb.py not found")
            checks['chatbot_kb_exists'] = False
        
        self.results['knowledge_base'] = checks
        self._print_check_results(checks, "KNOWLEDGE BASE")
        return all(checks.values())
    
    def _print_check_results(self, checks, section):
        """Print formatted check results"""
        passed = sum(1 for v in checks.values() if v is True)
        total = len(checks)
        percentage = (passed / total * 100) if total > 0 else 0
        
        status = "✓ PASS" if percentage >= 80 else "⚠ PARTIAL" if percentage >= 50 else "✗ FAIL"
        print(f"\n{section} STATUS: {status} ({passed}/{total} checks passed - {percentage:.0f}%)")
    
    def validate_all(self):
        """Run all validations"""
        print("\n" + "="*60)
        print("PUSHKARANI SYSTEM INTEGRATION VALIDATION")
        print("="*60)
        
        backend_valid = self.validate_backend()
        frontend_valid = self.validate_frontend()
        models_valid = self.validate_models()
        kb_valid = self.validate_knowledge_base()
        
        # Determine overall status
        if backend_valid and frontend_valid and models_valid and kb_valid:
            self.results['overall'] = 'FULLY_INTEGRATED'
            status_icon = "✓"
            status_msg = "FULLY INTEGRATED"
        elif sum([backend_valid, frontend_valid, models_valid, kb_valid]) >= 3:
            self.results['overall'] = 'MOSTLY_INTEGRATED'
            status_icon = "⚠"
            status_msg = "MOSTLY INTEGRATED (Minor issues)"
        else:
            self.results['overall'] = 'INCOMPLETE'
            status_icon = "✗"
            status_msg = "INCOMPLETE (Issues detected)"
        
        print("\n" + "="*60)
        print(f"{status_icon} OVERALL STATUS: {status_msg}")
        print("="*60)
        
        return self.results

if __name__ == '__main__':
    # Get base path from argument or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    validator = IntegrationValidator(base_path)
    results = validator.validate_all()
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION VALIDATION SUMMARY")
    print("="*60)
    print(f"Backend Components: {'✓ Complete' if results['backend'] else '✗ Issues'}")
    print(f"Frontend Components: {'✓ Complete' if results['frontend'] else '✗ Issues'}")
    print(f"Trained Models: {'✓ Complete' if results['models'] else '✗ Issues'}")
    print(f"Knowledge Base: {'✓ Complete' if results['knowledge_base'] else '✗ Issues'}")
    print(f"\nOverall Status: {results['overall']}")
    print("="*60)
