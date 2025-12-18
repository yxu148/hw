import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from src.analyzer.code_analyzer import CodeContext

logger = logging.getLogger(__name__)

@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair with metadata."""
    question: str
    answer: str
    context: Dict[str, Any]
    question_type: str
    difficulty: str  # 'easy', 'medium', 'hard'
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "metadata": self.metadata or {}
        }

class QAGenerator:
    """Generates questions and answers from code contexts."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.question_templates = self._load_question_templates()
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different code elements."""
        return {
            "function": [
                "What does the function {function_name} do?",
                "How does {function_name} work?",
                "What parameters does {function_name} accept?",
                "What is the return type of {function_name}?",
                "What is the time complexity of {function_name}?",
                "What are the edge cases for {function_name}?",
                "How would you test {function_name}?",
                "What are the side effects of calling {function_name}?",
            ],
            "class": [
                "What is the purpose of the {class_name} class?",
                "What are the main responsibilities of the {class_name} class?",
                "What design patterns are used in the {class_name} class?",
                "What are the dependencies of the {class_name} class?",
                "How would you extend the {class_name} class?",
                "What are the main methods of the {class_name} class?",
                "How does the {class_name} class interact with other components?",
            ],
            "code_block": [
                "What does this code do?\n\n{code}",
                "How would you refactor this code?\n\n{code}",
                "What are the potential issues with this code?\n\n{code}",
                "How would you optimize this code?\n\n{code}",
                "What is the purpose of this code block?\n\n{code}",
            ],
            "import": [
                "What is the purpose of importing {import_name}?",
                "How is {import_name} used in this codebase?",
                "What functionality does {import_name} provide?",
            ]
        }
    
    def generate_qa_pairs(self, contexts: List[CodeContext]) -> List[QuestionAnswerPair]:
        """Generate Q&A pairs from a list of code contexts."""
        qa_pairs = []
        
        for context in contexts:
            try:
                # Generate questions based on context type
                if context.function_name:
                    qa_pairs.extend(self._generate_function_qa(context))
                elif context.class_name:
                    qa_pairs.extend(self._generate_class_qa(context))
                else:
                    qa_pairs.extend(self._generate_generic_qa(context))
                
                # Generate import-related questions
                qa_pairs.extend(self._generate_import_qa(context))
                
            except Exception as e:
                logger.error(f"Error generating QA for {context.file_path}:{context.start_line}: {e}")
        
        return qa_pairs
    
    def _generate_function_qa(self, context: CodeContext) -> List[QuestionAnswerPair]:
        """Generate Q&A pairs for a function context."""
        pairs = []
        
        # Prepare template variables
        template_vars = {
            'function_name': context.function_name,
            'class_name': context.class_name or 'the function',
            'file_path': context.file_path
        }
        
        # Select random templates
        templates = random.sample(
            self.question_templates['function'],
            min(3, len(self.question_templates['function']))
        )
        
        for template in templates:
            try:
                question = template.format(**template_vars)
                answer = self._generate_function_answer(context, template)
                
                if answer:
                    pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=answer,
                        context=context.to_dict(),
                        question_type='function',
                        difficulty=random.choice(['easy', 'medium', 'hard'])
                    ))
            except Exception as e:
                logger.warning(f"Error generating function QA: {e}")
        
        return pairs
    
    def _generate_class_qa(self, context: CodeContext) -> List[QuestionAnswerPair]:
        """Generate Q&A pairs for a class context."""
        pairs = []
        
        if not context.class_name:
            return pairs
            
        template_vars = {
            'class_name': context.class_name,
            'file_path': context.file_path
        }
        
        templates = random.sample(
            self.question_templates['class'],
            min(2, len(self.question_templates['class']))
        )
        
        for template in templates:
            try:
                question = template.format(**template_vars)
                answer = self._generate_class_answer(context, template)
                
                if answer:
                    pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=answer,
                        context=context.to_dict(),
                        question_type='class',
                        difficulty=random.choice(['medium', 'hard'])
                    ))
            except Exception as e:
                logger.warning(f"Error generating class QA: {e}")
        
        return pairs
    
    def _generate_generic_qa(self, context: CodeContext) -> List[QuestionAnswerPair]:
        """Generate generic Q&A pairs for any code context."""
        pairs = []
        
        # Only use a small portion of code to avoid too long questions
        code_lines = context.code.split('\n')
        if len(code_lines) > 10:
            code_snippet = '\n'.join(code_lines[:10]) + '\n...'
        else:
            code_snippet = context.code
        
        template_vars = {
            'code': code_snippet,
            'file_path': context.file_path,
            'start_line': context.start_line,
            'end_line': context.end_line
        }
        
        templates = random.sample(
            self.question_templates['code_block'],
            min(2, len(self.question_templates['code_block']))
        )
        
        for template in templates:
            try:
                question = template.format(**template_vars)
                answer = self._generate_generic_answer(context, template)
                
                if answer:
                    pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=answer,
                        context=context.to_dict(),
                        question_type='code_block',
                        difficulty=random.choice(['easy', 'medium'])
                    ))
            except Exception as e:
                logger.warning(f"Error generating generic QA: {e}")
        
        return pairs
    
    def _generate_import_qa(self, context: CodeContext) -> List[QuestionAnswerPair]:
        """Generate Q&A pairs for imports in the context."""
        pairs = []
        
        if not context.imports:
            return pairs
        
        # Select a few random imports to generate questions for
        selected_imports = random.sample(
            context.imports,
            min(2, len(context.imports))
        )
        
        for imp in selected_imports:
            try:
                # Extract the main import name (simplified)
                import_name = imp.split()[1].split('.')[0] if ' ' in imp else imp.split('.')[0]
                
                template_vars = {
                    'import_name': import_name,
                    'full_import': imp,
                    'file_path': context.file_path
                }
                
                template = random.choice(self.question_templates['import'])
                question = template.format(**template_vars)
                answer = self._generate_import_answer(imp, context)
                
                if answer:
                    pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=answer,
                        context={
                            'import_statement': imp,
                            'file_path': context.file_path,
                            'line_range': f"{context.start_line}-{context.end_line}"
                        },
                        question_type='import',
                        difficulty='easy'
                    ))
            except Exception as e:
                logger.warning(f"Error generating import QA for {imp}: {e}")
        
        return pairs
    
    def _generate_function_answer(self, context: CodeContext, question_template: str) -> str:
        """Generate an answer for a function-related question."""
        # This is a simplified version - in practice, you'd want to use an LLM
        # or more sophisticated analysis to generate better answers
        
        answer_parts = []
        
        # Add docstring if available
        if context.docstring:
            answer_parts.append(f"This function is documented as: {context.docstring}")
        
        # Add code context
        answer_parts.append(f"Here's the implementation:\n```{self._get_language(context.file_path)}\n{context.code}\n```")
        
        # Add information about parameters and return values
        if context.code:
            answer_parts.append("The function appears to be defined with the following signature:")
            answer_parts.append(f"- File: {context.file_path}")
            if context.class_name:
                answer_parts.append(f"- Class: {context.class_name}")
            answer_parts.append(f"- Function: {context.function_name}")
        
        return '\n\n'.join(answer_parts)
    
    def _generate_class_answer(self, context: CodeContext, question_template: str) -> str:
        """Generate an answer for a class-related question."""
        answer_parts = []
        
        if context.docstring:
            answer_parts.append(f"Class documentation:\n{context.docstring}")
        
        answer_parts.append(f"Class definition from {context.file_path}:\n```{self._get_language(context.file_path)}\n{context.code}\n```")
        
        return '\n\n'.join(answer_parts)
    
    def _generate_generic_answer(self, context: CodeContext, question_template: str) -> str:
        """Generate an answer for a generic code block question."""
        language = self._get_language(context.file_path)
        
        answer = f"Here's the code block from {context.file_path} (lines {context.start_line}-{context.end_line}):"
        answer += f"\n\n```{language}\n{context.code}\n```"
        
        if 'refactor' in question_template.lower():
            answer += "\n\nA potential refactoring could include:"
            answer += "\n1. Extracting complex logic into well-named helper functions"
            answer += "\n2. Adding input validation and error handling"
            answer += "\n3. Improving variable names for better readability"
        
        return answer
    
    def _generate_import_answer(self, import_statement: str, context: CodeContext) -> str:
        """Generate an answer for an import-related question."""
        # This is a simplified version - in practice, you'd want to analyze the actual usage
        
        if ' from ' in import_statement:
            # Handle 'from x import y' style imports
            _, module = import_statement.split(' from ')
            module = module.split(' import ')[0].strip()
        else:
            # Handle 'import x' style imports
            module = import_statement.replace('import', '').strip()
        
        answer = f"The import statement `{import_statement}` is used in {context.file_path} "
        answer += f"(lines {context.start_line}-{context.end_line}).\n\n"
        
        # Add common information about well-known modules
        common_modules = {
            'os': 'provides a way of using operating system dependent functionality',
            'sys': 'provides access to some variables used or maintained by the interpreter',
            'json': 'implements a JSON encoder/decoder',
            're': 'provides regular expression matching operations',
            'datetime': 'supplies classes for manipulating dates and times',
            'logging': 'implements a flexible event logging system',
            'unittest': 'a unit testing framework',
            'pytest': 'a testing framework',
            'numpy': 'a fundamental package for scientific computing',
            'pandas': 'provides high-performance data structures and data analysis tools',
            'tensorflow': 'an end-to-end open source platform for machine learning',
            'torch': 'a deep learning framework',
            'flask': 'a lightweight WSGI web application framework',
            'django': 'a high-level Python Web framework',
            'fastapi': 'a modern, fast web framework for building APIs',
        }
        
        if module in common_modules:
            answer += f"The `{module}` module {common_modules[module]}. "
            answer += "It's commonly used for "
            
            if module in ['os', 'sys']:
                answer += "system-level operations and interacting with the operating system."
            elif module in ['json', 're']:
                answer += "data processing and manipulation."
            elif module in ['datetime', 'time']:
                answer += "handling dates, times, and time-related operations."
            elif module in ['unittest', 'pytest']:
                answer += "writing and running tests for Python code."
            elif module in ['numpy', 'pandas']:
                answer += "scientific computing and data analysis."
            elif module in ['tensorflow', 'torch']:
                answer += "machine learning and deep learning applications."
            elif module in ['flask', 'django', 'fastapi']:
                answer += "web development and building APIs."
            else:
                answer += "various programming tasks."
        else:
            answer += f"This appears to be a {module} module. "
            answer += "It might be a third-party package or a local module in the project. "
            answer += "The specific functionality would depend on how it's used in the code."
        
        return answer
    
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
        return language_map.get(ext, '')
