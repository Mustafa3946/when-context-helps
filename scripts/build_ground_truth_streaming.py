"""
Generate Ground Truth Using Streaming (INFOCOM-Compliant)

CLI entrypoint for streaming ground truth generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
from src.evaluation.ground_truth_streaming import StreamingGroundTruthBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ground truth using streaming (INFOCOM-compliant)"
    )
    parser.add_argument(
        '--parsed-logs',
        default='data/processed/bgl/parsed.jsonl',
        help='Path to parsed.jsonl'
    )
    parser.add_argument(
        '--template-map',
        default='data/processed/bgl/template_map.jsonl',
        help='Path to template_map.jsonl'
    )
    parser.add_argument(
        '--output',
        default='data/evaluation/bgl_ground_truth.jsonl',
        help='Output ground truth file (will also create _incidents.jsonl and _qrels.jsonl)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-incidents',
        type=int,
        default=100,
        help='Maximum incidents to generate'
    )
    parser.add_argument(
        '--min-relevant',
        type=int,
        default=3,
        help='Minimum relevant templates per incident'
    )
    parser.add_argument(
        '--time-window',
        type=float,
        default=1.0,
        help='Time window in hours for candidate filtering'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    builder = StreamingGroundTruthBuilder(
        parsed_logs_file=args.parsed_logs,
        template_map_db=args.template_map,
        output_file=args.output,
        seed=args.seed,
        max_incidents=args.max_incidents,
        min_relevant=args.min_relevant,
        time_window_hours=args.time_window
    )
    
    builder.build()
