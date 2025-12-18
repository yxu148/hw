# Codebase-Aware Training Data Generator

A system for generating high-quality training data from code repositories to fine-tune LLMs for code understanding and design tasks.

## Features

- Automated Q&A pair generation from codebases
- Design pattern extraction and documentation
- Support for multiple programming languages
- Context-aware data generation
- Customizable output formats
- Python 3.7 to 3.11 compatibility

## System Requirements

- Python 3.7 to 3.11 (recommended: Python 3.9+)
- Dependencies listed in `requirements.txt`

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `config.yaml` with your repository paths
4. Run `python process_repo.py`

## Test Repositories

Two test repositories are provided for demonstration:

1. `code_repo1/` - A small Python project demonstrating basic design patterns
2. `code_repo2/LightX2V-main` - A more complex codebase for testing

## Output Structure

Generated training data is saved in the `data/` directory with the following structure:
```
data/
├── <repo1_name>_qa.jsonl         # Generated Q&A pairs
├── <repo1_name>_patterns.json   # Extracted design patterns
├── <repo2_name>_qa.jsonl
├── <repo2_name>_patterns.json
└── generation_summary.json      # Summary of the generation process
```

## Training Data Structure

### Q&A Pairs Format (`*_qa.jsonl`)
Each line is a JSON object representing a single Q&A pair:
```json
{
  "question": "What is the purpose of function X?",
  "answer": "Function X is used to...",
  "context": {
    "file_path": "path/to/file.py",
    "function_name": "function_x",
    "code_snippet": "def function_x(...): ...",
    "imports": ["os", "sys"],
    "language": "python"
  },
  "metadata": {
    "generated_at": "2023-12-18T21:00:00",
    "confidence_score": 0.95,
    "difficulty": "intermediate"
  }
}
```

### Design Patterns Format (`*_patterns.json`)
```json
{
  "design_patterns": [
    {
      "pattern_name": "Singleton",
      "file_path": "path/to/singleton.py",
      "line_numbers": [10, 20],
      "confidence": 0.92,
      "code_snippet": "class Singleton:\n    _instance = None\n    ...",
      "description": "Ensures a class has only one instance..."
    }
  ],
  "architecture": {
    "components": [...],
    "dependencies": [...],
    "patterns_detected": ["Singleton", "Factory", ...]
  }
}
```

### Ensuring Data Quality and Diversity

1. **Code Coverage**:
   - Analyzes all code files in the repository
   - Processes different types of code elements (functions, classes, modules)
   - Handles various programming language features

2. **Question Diversity**:
   - Multiple question types (what, why, how, when)
   - Different complexity levels (basic to advanced)
   - Context-aware question generation

3. **Representative Sampling**:
   - Balances questions across different modules/packages
   - Ensures coverage of both simple and complex code patterns
   - Includes examples of both good and bad practices

4. **Validation**:
   - Syntax validation of generated code snippets
   - Semantic validation of Q&A pairs
   - Pattern matching for common error cases

## Project Structure

```
codebase-trainer/
├── src/
│   ├── analyzer/          # Code analysis modules
│   ├── generators/        # Q&A and design pattern generators
│   ├── validators/        # Data validation
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Output directory for generated datasets
└── tests/                 # Test cases
```
