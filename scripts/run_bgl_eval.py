"""
Run BGL evaluation experiments.

Usage:
    python scripts/run_bgl_eval.py --config config/bgl_eval.yaml
    
    python scripts/run_bgl_eval.py --config config/bgl_eval.yaml --max-queries 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.run_experiments import ExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run LAR-RAG evaluation experiments on BGL dataset"
    )
    
    parser.add_argument(
        "--config",
        default="config/bgl_eval.yaml",
        help="Path to experiment configuration YAML"
    )
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Specific methods to run (default: all from config)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--run-id",
        default=None,
        help="Custom run ID (default: auto-generate)"
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Initialize experiment runner
        logger.info(f"Loading configuration from {config_path}")
        runner = ExperimentRunner(
            config_path=str(config_path),
            max_queries=args.max_queries,
            output_dir=args.output_dir,
            run_id=args.run_id
        )
        
        # Run experiments
        logger.info("=" * 60)
        logger.info("Starting LAR-RAG Evaluation")
        logger.info("=" * 60)
        
        results = runner.run_all_methods(method_filter=args.methods)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info("=" * 60)
        
        logger.info(f"Results saved to: {runner.output_dir}")
        logger.info(f"Summary CSV: {runner.output_dir / 'summary.csv'}")
        
        # Print quick metrics table
        logger.info("\nQuick Results:")
        logger.info("-" * 60)
        logger.info(f"{'Method':<25} {'MRR':>10} {'HR@5':>10} {'Latency (ms)':>15}")
        logger.info("-" * 60)
        
        for method_name, result in results.items():
            logger.info(
                f"{method_name:<25} "
                f"{result.mrr:>10.4f} "
                f"{result.hr_at_5:>10.4f} "
                f"{result.latency_median_ms:>15.2f}"
            )
        
        logger.info("-" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
