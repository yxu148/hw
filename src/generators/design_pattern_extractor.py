import ast
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DesignPattern:
    """Represents a design pattern found in the codebase."""
    pattern_type: str
    description: str
    location: str
    confidence: float  # 0.0 to 1.0
    code_snippets: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DesignPatternExtractor:
    """Extracts and documents design patterns from code."""
    
    def __init__(self):
        self.pattern_definitions = self._load_pattern_definitions()
    
    def _load_pattern_definitions(self) -> Dict[str, Dict]:
        """Load pattern definitions and their detection rules."""
        return {
            'singleton': {
                'description': 'Ensures a class has only one instance and provides a global point of access to it.',
                'indicators': [
                    r'__new__.*\bcls\._instance\b',
                    r'@classmethod\s+def\s+get_instance\s*\('
                ],
                'confidence_threshold': 0.7,
                'language': 'python'
            },
            'factory': {
                'description': 'Defines an interface for creating an object, but lets subclasses alter the type of objects that will be created.',
                'indicators': [
                    r'def\s+create_\w+\s*\(',
                    r'class\s+\w+Factory\b',
                    r'def\s+make_\w+\s*\('
                ],
                'confidence_threshold': 0.6,
                'language': 'any'
            },
            'observer': {
                'description': 'Lets you define a subscription mechanism to notify multiple objects about any events that happen to the object they\'re observing.',
                'indicators': [
                    r'\bnotify_observers\s*\(',
                    r'\badd_observer\s*\(',
                    r'\bremove_observer\s*\('
                ],
                'confidence_threshold': 0.7,
                'language': 'any'
            },
            'strategy': {
                'description': 'Defines a family of algorithms, puts each of them into a separate class, and makes their objects interchangeable.',
                'indicators': [
                    r'class\s+\w+Strategy\b',
                    r'def\s+execute\s*\(',
                    r'def\s+apply_strategy\s*\('
                ],
                'confidence_threshold': 0.65,
                'language': 'any'
            },
            'decorator': {
                'description': 'Lets you attach new behaviors to objects by placing these objects inside special wrapper objects that contain the behaviors.',
                'indicators': [
                    r'def\s+\w+_decorator\s*\(',
                    r'@\w+\s*\n\s*def',
                    r'def\s+decorator\s*\('
                ],
                'confidence_threshold': 0.8,
                'language': 'python'
            },
            'repository': {
                'description': 'Mediates between the domain and data mapping layers, acting like an in-memory collection of domain objects.',
                'indicators': [
                    r'class\s+\w+Repository\b',
                    r'def\s+get_\w+_by_id\s*\(',
                    r'def\s+save_\w+\s*\('
                ],
                'confidence_threshold': 0.7,
                'language': 'any'
            },
            'dependency_injection': {
                'description': 'A technique where one object supplies the dependencies of another object.',
                'indicators': [
                    r'def\s+__init__\s*\([^)]*\b\w+\s*:\s*\w+[^)]*\)',
                    r'@inject',
                    r'def\s+provide_\w+\s*\('
                ],
                'confidence_threshold': 0.75,
                'language': 'any'
            },
            'builder': {
                'description': 'Lets you construct complex objects step by step.',
                'indicators': [
                    r'class\s+\w+Builder\b',
                    r'def\s+with_\w+\s*\('
                ],
                'confidence_threshold': 0.7,
                'language': 'any'
            },
            'adapter': {
                'description': 'Allows objects with incompatible interfaces to collaborate.',
                'indicators': [
                    r'class\s+\w+Adapter\b',
                    r'class\s+\w+\s*\(.*Adapter\)'
                ],
                'confidence_threshold': 0.8,
                'language': 'any'
            },
            'facade': {
                'description': 'Provides a simplified interface to a library, a framework, or any other complex set of classes.',
                'indicators': [
                    r'class\s+\w+Facade\b',
                    r'def\s+simplified_\w+\s*\('
                ],
                'confidence_threshold': 0.7,
                'language': 'any'
            }
        }
    
    def extract_patterns_from_file(self, file_path: str, content: str) -> List[DesignPattern]:
        """Extract design patterns from a single file."""
        patterns = []
        
        # Get file extension to determine language
        language = self._get_language(file_path)
        
        # Check for each pattern definition
        for pattern_name, pattern_def in self.pattern_definitions.items():
            # Skip if language doesn't match (unless pattern is language-agnostic)
            if pattern_def['language'] != 'any' and pattern_def['language'] != language:
                continue
                
            # Check each indicator for this pattern
            matches = []
            for indicator in pattern_def['indicators']:
                if re.search(indicator, content, re.MULTILINE | re.DOTALL):
                    matches.append(indicator)
            
            # Calculate confidence based on number of matches
            if matches:
                confidence = min(1.0, len(matches) * 0.3)  # Cap at 1.0
                if confidence >= pattern_def['confidence_threshold']:
                    # Get code snippets that matched
                    code_snippets = []
                    for match in matches[:3]:  # Limit to 3 snippets per pattern
                        # Find the matching lines
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(match, line):
                                # Get some context around the match
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = '\n'.join(lines[start:end])
                                code_snippets.append(snippet)
                                break  # Just get the first match per indicator
                    
                    patterns.append(DesignPattern(
                        pattern_type=pattern_name,
                        description=pattern_def['description'],
                        location=file_path,
                        confidence=round(confidence, 2),
                        code_snippets=code_snippets
                    ))
        
        return patterns
    
    def analyze_architecture(self, codebase_structure: Dict) -> Dict[str, Any]:
        """Analyze the overall architecture of the codebase."""
        architecture = {
            'layers': self._identify_architecture_layers(codebase_structure),
            'patterns': [],
            'module_dependencies': {},
            'entry_points': []
        }
        
        # Identify patterns across the entire codebase
        all_patterns = []
        for file_path, contexts in codebase_structure.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                patterns = self.extract_patterns_from_file(file_path, content)
                all_patterns.extend(patterns)
        
        architecture['patterns'] = [p.__dict__ for p in all_patterns]
        
        # Analyze module dependencies
        architecture['module_dependencies'] = self._analyze_dependencies(codebase_structure)
        
        # Identify entry points (files that might be executed directly)
        architecture['entry_points'] = self._identify_entry_points(codebase_structure)
        
        return architecture
    
    def _identify_architecture_layers(self, codebase_structure: Dict) -> List[Dict]:
        """Identify architectural layers based on directory structure and imports."""
        layers = []
        
        # Common layer patterns
        layer_patterns = {
            'presentation': ['view', 'ui', 'templates', 'static', 'components'],
            'application': ['service', 'usecase', 'interactor', 'application'],
            'domain': ['model', 'entity', 'domain', 'business'],
            'infrastructure': ['repository', 'dao', 'persistence', 'db', 'database', 'external']
        }
        
        # Count occurrences of layer indicators in file paths
        layer_counts = {layer: 0 for layer in layer_patterns}
        
        for file_path in codebase_structure.keys():
            path_lower = str(file_path).lower()
            for layer, patterns in layer_patterns.items():
                if any(pattern in path_lower for pattern in patterns):
                    layer_counts[layer] += 1
        
        # Create layer information
        for layer, count in layer_counts.items():
            if count > 0:  # Only include layers that have matching files
                layers.append({
                    'name': layer,
                    'description': self._get_layer_description(layer),
                    'file_count': count
                })
        
        return layers
    
    def _get_layer_description(self, layer: str) -> str:
        """Get a description for an architectural layer."""
        descriptions = {
            'presentation': 'Handles user interface and user interaction',
            'application': 'Contains application-specific business rules and use cases',
            'domain': 'Contains enterprise-wide business rules and entities',
            'infrastructure': 'Provides technical capabilities to support higher layers'
        }
        return descriptions.get(layer, 'Unknown layer')
    
    def _analyze_dependencies(self, codebase_structure: Dict) -> Dict[str, List[str]]:
        """Analyze dependencies between modules."""
        dependencies = {}
        
        for file_path, contexts in codebase_structure.items():
            for context in contexts:
                if context.imports:
                    dependencies[file_path] = context.imports
        
        return dependencies
    
    def _identify_entry_points(self, codebase_structure: Dict) -> List[str]:
        """Identify potential entry points in the codebase."""
        entry_points = []
        
        for file_path in codebase_structure.keys():
            path = Path(file_path)
            if path.name == 'main.py' or path.name == 'app.py' or path.name == 'manage.py':
                entry_points.append(str(path))
            elif path.name.endswith('__main__.py'):
                entry_points.append(str(path))
            elif path.suffix == '.py' and path.stem == path.parent.name:
                # Common pattern for package entry points
                entry_points.append(str(path))
        
        return entry_points
    
    def _get_language(self, file_path: str) -> str:
        """Get the programming language from the file extension."""
        ext = file_path.split('.')[-1].lower()
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php',
            'c': 'c',
            'h': 'c',
            'cpp': 'cpp',
            'hpp': 'cpp',
            'cs': 'csharp',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala',
            'm': 'objective-c',
            'mm': 'objective-c++',
        }
        return language_map.get(ext, 'unknown')
