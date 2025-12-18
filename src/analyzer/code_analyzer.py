import os
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tree_sitter
from tree_sitter import Language, Parser
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeContext:
    """Represents the context of a code snippet."""
    file_path: str
    start_line: int
    end_line: int
    code: str
    imports: List[str]
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    docstring: Optional[str] = None
    variables: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "imports": self.imports,
            "class_name": self.class_name,
            "function_name": self.function_name,
            "docstring": self.docstring,
            "variables": self.variables or []
        }

class CodeAnalyzer:
    """Analyzes code files and extracts structural information."""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.parser = self._initialize_parser()
    
    def _initialize_parser(self) -> Parser:
        """Initialize the tree-sitter parser for the specified language."""
        # This is a simplified version - in production, you'd want to build the language library
        parser = Parser()
        try:
            # Try to load the language library
            Language.build_library(
                'build/my-languages.so',
                [
                    'vendor/tree-sitter-python',
                    'vendor/tree-sitter-javascript',
                    'vendor/tree-sitter-java'
                ]
            )
            PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
            parser.set_language(PYTHON_LANGUAGE)
        except Exception as e:
            logger.warning(f"Could not load tree-sitter language: {e}")
            parser = None
        return parser
    
    def analyze_file(self, file_path: str) -> List[CodeContext]:
        """Analyze a single file and return code contexts."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.language == 'python':
                return self._analyze_python_file(file_path, content)
            # Add other language analyzers here
            else:
                return self._analyze_generic_file(file_path, content)
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return []
    
    def _analyze_python_file(self, file_path: str, content: str) -> List[CodeContext]:
        """Analyze a Python file using AST."""
        contexts = []
        try:
            tree = ast.parse(content)
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    context = self._extract_python_context(node, file_path, content)
                    if context:
                        contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            return []
    
    def _extract_python_context(
        self, 
        node: ast.AST, 
        file_path: str, 
        content: str
    ) -> Optional[CodeContext]:
        """Extract context from a Python AST node."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Get the code snippet
            lines = content.splitlines()
            code_snippet = '\n'.join(lines[start_line-1:end_line])
            
            # Extract docstring
            docstring = ast.get_docstring(node, clean=True)
            
            # Get imports (simplified - in practice, you'd want to track scope)
            imports = []
            for n in ast.walk(node):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(n))
            
            # Get variable names (simplified)
            variables = []
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    variables.append(n.id)
            
            return CodeContext(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code_snippet,
                imports=imports,
                class_name=getattr(node, 'name', None) if isinstance(node, ast.ClassDef) else None,
                function_name=getattr(node, 'name', None) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                docstring=docstring,
                variables=list(set(variables))  # Remove duplicates
            )
            
        except Exception as e:
            logger.error(f"Error extracting context from node: {e}")
            return None
    
    def _analyze_generic_file(self, file_path: str, content: str) -> List[CodeContext]:
        """Fallback analyzer for unsupported languages."""
        # This is a simplified version - in practice, you'd want to implement
        # language-specific parsers or use tree-sitter for better results
        return [
            CodeContext(
                file_path=file_path,
                start_line=1,
                end_line=len(content.splitlines()),
                code=content,
                imports=[],
                docstring=None
            )
        ]

class RepositoryAnalyzer:
    """Analyzes an entire code repository."""
    
    def __init__(self, repo_path: str, languages: List[str] = None):
        self.repo_path = Path(repo_path)
        self.languages = languages or ["python"]
        self.analyzers = {lang: CodeAnalyzer(lang) for lang in self.languages}
    
    def analyze(self) -> Dict[str, List[CodeContext]]:
        """Analyze the entire repository."""
        results = {}
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = Path(root) / file
                language = self._detect_language(file_path)
                
                if language in self.analyzers:
                    contexts = self.analyzers[language].analyze_file(str(file_path))
                    if contexts:
                        results[str(file_path)] = contexts
        
        return results
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming language of a file based on its extension."""
        ext = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.c': 'c',
            '.h': 'c',
            '.cpp': 'cpp',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.m': 'objective-c',
            '.mm': 'objective-c++',
        }
        
        return language_map.get(ext)
