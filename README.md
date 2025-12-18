# Codebase-Aware Training Data Generator

A system for generating high-quality training data from code repositories to fine-tune LLMs for code understanding and design tasks.

## Features

- Automated Q&A pair generation from codebases
- Design pattern extraction and documentation
- Support for multiple programming languages
- Context-aware data generation
- Customizable output formats

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `config.yaml` with your repository paths
4. Run `python main.py`

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
