import os
import sys
import yaml
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from analyzer.code_analyzer import RepositoryAnalyzer, CodeContext
from generators.qa_generator import QAGenerator, QuestionAnswerPair
from generators.design_pattern_extractor import DesignPatternExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """Main class for generating training data from local code repositories."""
    
    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        """Initialize with configuration.
        
        Args:
            config_path: Path to the configuration file. Defaults to "config/config.yaml".
            
        Raises:
            KeyError: If required configuration keys are missing.
            Exception: For any other errors during initialization.
        """
        try:
            self.config = self._load_config(config_path)
            
            # Validate required configuration sections
            required_sections = ['output', 'repositories']
            for section in required_sections:
                if section not in self.config:
                    error_msg = f"Missing required configuration section: {section}"
                    logger.error(error_msg)
                    raise KeyError(error_msg)
            
            # Set up output directory
            self.output_dir = Path(self.config['output'].get('output_dir', 'data'))
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                error_msg = f"Failed to create output directory {self.output_dir}: {e}"
                logger.error(error_msg)
                raise OSError(error_msg) from e
            
            # Initialize components with error handling
            self.qa_generator = QAGenerator(self.config.get('qa_generation', {}))
            self.design_extractor = DesignPatternExtractor()
            
            logger.info("TrainingDataGenerator initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize TrainingDataGenerator: {e}")
            raise
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dict containing the loaded configuration.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If there's an error parsing the YAML.
            Exception: For any other errors.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    error_msg = f"Invalid configuration format in {config_path}. Expected a dictionary."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                return config
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML in {config_path}: {e}"
            logger.error(error_msg)
            raise yaml.YAMLError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading config from {config_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def process_repository(self, repo_config: Dict) -> Dict[str, Any]:
        """Process a single local repository configuration.
        
        Args:
            repo_config: Dictionary containing repository configuration.
                Expected keys: 'path' (str), 'languages' (List[str], optional)
                
        Returns:
            Dictionary containing analysis results with keys:
            - repository: Path to the repository
            - qa_pairs: List of generated QA pairs
            - design_patterns: List of detected design patterns
            - architecture: Dictionary of architectural analysis
            - stats: Dictionary of processing statistics
            - errors: List of error messages if any occurred
        """
        result = {
            'repository': '',
            'qa_pairs': [],
            'design_patterns': [],
            'architecture': {},
            'stats': {
                'files_processed': 0,
                'files_failed': 0,
                'qa_pairs_generated': 0,
                'patterns_found': 0,
                'start_time': datetime.datetime.now().isoformat()
            },
            'errors': []
        }
        
        try:
            # Validate repository path
            if 'path' not in repo_config:
                error_msg = "Repository configuration missing required 'path' key"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                return result
                
            repo_path = Path(repo_config['path']).resolve()
            result['repository'] = str(repo_path)
            
            if not repo_path.exists():
                error_msg = f"Repository path does not exist: {repo_path}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                return result
            
            if not repo_path.is_dir():
                error_msg = f"Repository path is not a directory: {repo_path}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                return result
            
            # Initialize analyzer with languages from config or default to Python
            languages = repo_config.get('languages', ['python'])
            logger.info(f"Analyzing repository at {repo_path} (languages: {', '.join(languages)})...")
            
            try:
                analyzer = RepositoryAnalyzer(str(repo_path), languages=languages)
                code_contexts = analyzer.analyze()
                result['stats']['files_processed'] = len(code_contexts)
                
                if not code_contexts:
                    logger.warning(f"No code files found in repository: {repo_path}")
                    return result
                
                # Process each file for QA generation and pattern extraction
                qa_pairs = []
                design_patterns = []
                
                for file_path, contexts in code_contexts.items():
                    try:
                        # Generate QA pairs
                        try:
                            file_qa_pairs = self.qa_generator.generate_qa_pairs(contexts)
                            qa_pairs.extend(file_qa_pairs)
                            result['stats']['qa_pairs_generated'] += len(file_qa_pairs)
                        except Exception as e:
                            error_msg = f"Error generating QA pairs for {file_path}: {e}"
                            logger.error(error_msg, exc_info=True)
                            result['errors'].append(error_msg)
                            result['stats']['files_failed'] += 1
                            continue
                        
                        # Extract design patterns
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                patterns = self.design_extractor.extract_patterns_from_file(file_path, content)
                                design_patterns.extend(patterns)
                                result['stats']['patterns_found'] += len(patterns)
                        except Exception as e:
                            error_msg = f"Error extracting design patterns from {file_path}: {e}"
                            logger.error(error_msg, exc_info=True)
                            result['errors'].append(error_msg)
                            continue
                            
                    except Exception as e:
                        error_msg = f"Unexpected error processing {file_path}: {e}"
                        logger.error(error_msg, exc_info=True)
                        result['errors'].append(error_msg)
                        result['stats']['files_failed'] += 1
                        continue
                
                # Analyze overall architecture if we have patterns
                if design_patterns:
                    try:
                        architecture = self.design_extractor.analyze_architecture(code_contexts)
                        result['architecture'] = architecture
                    except Exception as e:
                        error_msg = f"Error analyzing architecture: {e}"
                        logger.error(error_msg, exc_info=True)
                        result['errors'].append(error_msg)
                
                # Update result with successful processing
                result['qa_pairs'] = [asdict(qa) for qa in qa_pairs]
                result['design_patterns'] = [asdict(p) for p in design_patterns]
                result['stats']['end_time'] = datetime.datetime.now().isoformat()
                
                logger.info(
                    f"Repository processing complete. "
                    f"Files: {result['stats']['files_processed']} processed, "
                    f"{result['stats']['files_failed']} failed, "
                    f"{result['stats']['qa_pairs_generated']} QA pairs generated, "
                    f"{result['stats']['patterns_found']} design patterns found"
                )
                
                if result['errors']:
                    logger.warning(f"Completed with {len(result['errors'])} errors")
                
                return result
                
            except Exception as e:
                error_msg = f"Error initializing repository analyzer for {repo_path}: {e}"
                logger.error(error_msg, exc_info=True)
                result['errors'].append(error_msg)
                return result
                
        except Exception as e:
            error_msg = f"Unexpected error processing repository: {e}"
            logger.critical(error_msg, exc_info=True)
            result['errors'].append(error_msg)
            return result
    
    def save_results(self, results: Dict, output_path: Path) -> Dict[str, str]:
        """Save the generated data to files.
        
        Args:
            results: Dictionary containing the results to save
            output_path: Base path for output files
            
        Returns:
            Dictionary with paths to the saved files
            
        Raises:
            OSError: If there's an error writing to the filesystem
            json.JSONEncodeError: If there's an error encoding the data to JSON
        """
        saved_files = {}
        
        try:
            # Ensure output directory exists
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                error_msg = f"Failed to create output directory {output_path.parent}: {e}"
                logger.error(error_msg)
                raise OSError(error_msg) from e
            
            # Save QA pairs if any exist
            qa_pairs = results.get('qa_pairs', [])
            if qa_pairs:
                qa_path = output_path.with_name(f"{output_path.stem}_qa.jsonl")
                try:
                    with open(qa_path, 'w', encoding='utf-8') as f:
                        for qa in qa_pairs:
                            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
                    saved_files['qa_pairs'] = str(qa_path)
                    logger.info(f"Saved {len(qa_pairs)} QA pairs to {qa_path}")
                except (IOError, OSError, TypeError) as e:
                    error_msg = f"Failed to save QA pairs to {qa_path}: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise OSError(error_msg) from e
            
            # Save design patterns and architecture
            design_patterns = results.get('design_patterns', [])
            architecture = results.get('architecture', {})
            
            if design_patterns or architecture:
                patterns_path = output_path.with_name(f"{output_path.stem}_patterns.json")
                try:
                    with open(patterns_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            {
                                'design_patterns': design_patterns,
                                'architecture': architecture,
                                'metadata': {
                                    'generated_at': datetime.datetime.now().isoformat(),
                                    'total_patterns': len(design_patterns)
                                }
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                            default=str  # Handle datetime serialization
                        )
                    saved_files['design_patterns'] = str(patterns_path)
                    logger.info(f"Saved {len(design_patterns)} design patterns to {patterns_path}")
                except (IOError, OSError, TypeError) as e:
                    error_msg = f"Failed to save design patterns to {patterns_path}: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise OSError(error_msg) from e
            
            # Save errors if any exist
            errors = results.get('errors', [])
            if errors:
                errors_path = output_path.with_name(f"{output_path.stem}_errors.json")
                try:
                    with open(errors_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            {'errors': errors, 'count': len(errors)},
                            f,
                            indent=2,
                            ensure_ascii=False
                        )
                    saved_files['errors'] = str(errors_path)
                    logger.warning(f"Saved {len(errors)} errors to {errors_path}")
                except (IOError, OSError) as e:
                    logger.error(f"Failed to save errors to {errors_path}: {e}", exc_info=True)
                    # Don't re-raise for errors file as it's not critical
            
            return saved_files
            
        except Exception as e:
            logger.critical(f"Unexpected error in save_results: {e}", exc_info=True)
            raise
    
    def run(self) -> Dict[str, Any]:
        """Run the training data generation pipeline.
        
        Returns:
            Dictionary containing processing results and statistics
            
        Raises:
            ValueError: If no repositories are configured
            Exception: For any other unexpected errors
        """
        start_time = datetime.datetime.now()
        results = {
            'status': 'started',
            'start_time': start_time.isoformat(),
            'repositories': {},
            'statistics': {
                'total_repositories': 0,
                'successful_repositories': 0,
                'failed_repositories': 0,
                'total_qa_pairs': 0,
                'total_patterns': 0,
                'total_errors': 0
            },
            'configuration': {
                'config_path': str(Path(self.config.get('_config_path', 'unknown')).resolve()),
                'output_dir': str(self.output_dir)
            },
            'errors': []
        }
        
        try:
            # Validate repositories configuration
            if not self.config.get('repositories'):
                error_msg = "No repositories configured for processing"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['status'] = 'failed'
                raise ValueError(error_msg)
            
            results['statistics']['total_repositories'] = len(self.config['repositories'])
            logger.info(f"Starting processing of {results['statistics']['total_repositories']} repositories...")
            
            # Process each repository
            for repo_config in self.config['repositories']:
                repo_name = ""
                try:
                    repo_path = Path(repo_config.get('path', '')).resolve()
                    repo_name = repo_path.name if repo_path.exists() else 'unknown'
                    
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Processing repository: {repo_name}")
                    logger.info(f"Path: {repo_path}")
                    logger.info(f"Languages: {', '.join(repo_config.get('languages', ['python']))}")
                    logger.info(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info("="*50)
                    
                    # Process the repository
                    result = self.process_repository(repo_config)
                    
                    if not result or 'errors' not in result:
                        error_msg = f"Unexpected result format from process_repository for {repo_name}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                        results['statistics']['failed_repositories'] += 1
                        continue
                    
                    # Save results
                    output_base = self.output_dir / f"{repo_name}_training_data"
                    try:
                        saved_files = self.save_results(result, output_base)
                        
                        # Update statistics
                        repo_qa_count = len(result.get('qa_pairs', []))
                        repo_patterns_count = len(result.get('design_patterns', []))
                        repo_errors_count = len(result.get('errors', []))
                        
                        results['repositories'][repo_name] = {
                            'path': str(repo_path),
                            'status': 'completed' if not repo_errors_count else 'completed_with_errors',
                            'qa_pairs': repo_qa_count,
                            'design_patterns': repo_patterns_count,
                            'errors': repo_errors_count,
                            'output_files': saved_files,
                            'start_time': result.get('stats', {}).get('start_time'),
                            'end_time': result.get('stats', {}).get('end_time')
                        }
                        
                        results['statistics']['successful_repositories'] += 1
                        results['statistics']['total_qa_pairs'] += repo_qa_count
                        results['statistics']['total_patterns'] += repo_patterns_count
                        results['statistics']['total_errors'] += repo_errors_count
                        
                        logger.info(f"Successfully processed repository: {repo_name}")
                        
                    except Exception as save_error:
                        error_msg = f"Failed to save results for {repo_name}: {save_error}"
                        logger.error(error_msg, exc_info=True)
                        results['errors'].append(error_msg)
                        results['statistics']['failed_repositories'] += 1
                        
                except Exception as repo_error:
                    error_msg = f"Error processing repository {repo_name or 'unknown'}: {repo_error}"
                    logger.error(error_msg, exc_info=True)
                    results['errors'].append(error_msg)
                    results['statistics']['failed_repositories'] += 1
                    continue
            
            # Finalize results
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results.update({
                'status': 'completed',
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'duration_human_readable': str(end_time - start_time)
            })
            
            # Save summary
            summary_path = self.output_dir / "generation_summary.json"
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        results,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str  # Handle datetime serialization
                    )
                logger.info(f"\n{'-'*50}")
                logger.info(f"Training data generation complete!")
                logger.info(f"Summary saved to: {summary_path}")
                logger.info(f"Total repositories: {results['statistics']['total_repositories']}")
                logger.info(f"Successfully processed: {results['statistics']['successful_repositories']}")
                logger.info(f"Failed: {results['statistics']['failed_repositories']}")
                logger.info(f"Total QA pairs generated: {results['statistics']['total_qa_pairs']}")
                logger.info(f"Total design patterns found: {results['statistics']['total_patterns']}")
                logger.info(f"Total errors encountered: {results['statistics']['total_errors']}")
                logger.info(f"Duration: {results['duration_human_readable']}")
                logger.info(f"{'='*50}")
                
            except Exception as summary_error:
                logger.error(f"Failed to save generation summary: {summary_error}", exc_info=True)
                raise
            
            return results
            
        except Exception as e:
            error_msg = f"Fatal error in training data generation pipeline: {e}"
            logger.critical(error_msg, exc_info=True)
            results.update({
                'status': 'failed',
                'end_time': datetime.datetime.now().isoformat(),
                'error': str(e)
            })
            raise Exception(error_msg) from e

def main():
    """Main entry point for the script."""
    import argparse
    from datetime import datetime
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate training data from local code repositories.')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to the configuration file')
    parser.add_argument('--path', type=str, help='Path to the local repository')
    parser.add_argument('--languages', nargs='+', default=['python'],
                      help='Programming languages to analyze (default: python)')
    parser.add_argument('--output-dir', type=str, help='Output directory for generated data')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.path:
        config['repositories'] = [{
            'path': args.path,
            'languages': args.languages
        }]
    
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    # Run the generator
    logger.info("Starting training data generation...")
    generator = TrainingDataGenerator(config_path)
    results = {}
    
    try:
        results = generator.run()
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not results or 'repositories' not in results or not results['repositories']:
        print("\nNo repositories were processed successfully.")
        if 'errors' in results and results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"- {error}")
        return
    
    # Print repository results
    for repo_name, repo_data in results['repositories'].items():
        print(f"\nRepository: {repo_name}")
        print(f"Status: {repo_data.get('status', 'unknown')}")
        print(f"- QA Pairs: {repo_data.get('qa_pairs', 0)}")
        print(f"- Design Patterns: {repo_data.get('design_patterns', 0)}")
        
        output_files = repo_data.get('output_files', {})
        if output_files:
            print("Output Files:")
            for file_type, file_path in output_files.items():
                print(f"  - {file_type}: {file_path}")
    
    # Print summary statistics if available
    if 'statistics' in results:
        stats = results['statistics']
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total repositories: {stats.get('total_repositories', 0)}")
        print(f"Successfully processed: {stats.get('successful_repositories', 0)}")
        print(f"Failed: {stats.get('failed_repositories', 0)}")
        print(f"Total QA pairs generated: {stats.get('total_qa_pairs', 0)}")
        print(f"Total design patterns found: {stats.get('total_patterns', 0)}")
        print(f"Total errors: {stats.get('total_errors', 0)}")
        if 'duration_human_readable' in results:
            print(f"Duration: {results['duration_human_readable']}")

    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
