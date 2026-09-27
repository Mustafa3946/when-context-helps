"""
BGL Template Extraction and Deduplication

Extracts canonical log templates by replacing variable parts with placeholders.
This reduces embedding cost by only embedding unique templates instead of all logs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import hashlib
from collections import defaultdict
from typing import Dict, List
from utils.io_utils import read_json_or_jsonl


def extract_template(message: str) -> str:
    """
    Extract canonical template from BGL log message.
    
    Replaces variable parts with placeholders while preserving semantic signal:
    - Hex addresses (0x...) → <HEX>
    - UUIDs → <UUID>
    - IP addresses → <IP>
    - File paths (Unix/Windows) → <PATH>
    - Long integers (≥4 digits) → <NUM>
    - Timestamps → <TS>
    
    Preserves 1-3 digit numbers (often error codes).
    
    Args:
        message: Raw log message
        
    Returns:
        Template string with placeholders
    """
    template = message
    
    # 1. Replace hex addresses (0x...)
    template = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', template)
    
    # 2. Replace UUIDs
    template = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        '<UUID>',
        template
    )
    
    # 3. Replace IP addresses
    template = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', template)
    
    # 4. Replace file paths (Unix and Windows)
    template = re.sub(r'/[\w/\-\.]+', '<PATH>', template)  # Unix paths
    template = re.sub(r'[A-Z]:\\[\w\\\-\.]+', '<PATH>', template)  # Windows paths
    
    # 5. Replace long integers (4+ digits) - preserves error codes
    template = re.sub(r'\b\d{4,}\b', '<NUM>', template)
    
    # 6. Replace timestamps
    template = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', template)
    template = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', template)
    
    # 7. Normalize whitespace
    template = ' '.join(template.split())
    
    return template


def create_template_id(template: str) -> str:
    """
    Generate deterministic template ID from template text.
    
    Uses full SHA1 hash for collision resistance.
    
    Args:
        template: Template string
        
    Returns:
        Full SHA1 hex string (40 chars)
    """
    return hashlib.sha1(template.encode('utf-8')).hexdigest()


def extract_templates(
    input_file: str,
    templates_output: str,
    template_map_output: str,
    node_field: str = "node_id"
):
    """
    Extract templates from parsed BGL logs.
    
    STREAMING IMPLEMENTATION (memory-safe):
    - Reads logs line-by-line (no full load into memory)
    - Writes template_map incrementally
    - Accumulates only template metadata (not log_ids)
    
    Creates:
    1. templates.jsonl - Unique templates with counts
    2. template_map.jsonl - Mapping from logs to templates
    
    Args:
        input_file: Path to parsed BGL logs (JSON or JSONL)
        templates_output: Path to save templates.jsonl
        template_map_output: Path to save template_map.jsonl
        node_field: Field name for node ID (default: "node_id", fallback: "node")
    """
    print("="*80)
    print("BGL Template Extraction (Streaming)")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Templates output: {templates_output}")
    print(f"Template map output: {template_map_output}")
    print()
    
    # Prepare output directories
    Path(templates_output).parent.mkdir(parents=True, exist_ok=True)
    Path(template_map_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Template accumulator (only metadata, not log_ids)
    template_data = defaultdict(lambda: {"count": 0, "example_message": ""})
    
    # Counters
    total_logs = 0
    missing_node_count = 0
    
    # Stream logs and write template_map incrementally
    print("Streaming logs and writing template map...")
    
    with open(template_map_output, 'w', encoding='utf-8') as map_file:
        for log in read_json_or_jsonl(input_file):
            total_logs += 1
            
            # Validate and normalize fields
            if 'message' not in log or 'log_id' not in log:
                print(f"Warning: Skipping log without message or log_id: {log.get('log_id', 'unknown')}")
                continue
            
            message = log['message']
            
            # Handle node_id field (with fallback)
            node_id = log.get(node_field)
            if not node_id:
                # Try fallback
                node_id = log.get('node' if node_field == 'node_id' else 'node_id')
            if not node_id:
                node_id = "unknown"
                missing_node_count += 1
            
            # Extract template
            template = extract_template(message)
            template_id = create_template_id(template)
            
            # Update template data (accumulate metadata only, not log_ids)
            template_data[template_id]["count"] += 1
            if not template_data[template_id]["example_message"]:
                template_data[template_id]["example_message"] = message
            
            # Write template map record immediately (streaming)
            map_record = {
                "log_id": log['log_id'],
                "template_id": template_id,
                "timestamp": log.get('timestamp', ''),
                "node": node_id  # Use 'node' for consistency with parsed logs
            }
            map_file.write(json.dumps(map_record) + '\n')
            
            # Progress indicator
            if total_logs % 100000 == 0:
                print(f"  Processed {total_logs:,} logs, {len(template_data):,} unique templates...")
            # Progress indicator
            if total_logs % 100000 == 0:
                print(f"  Processed {total_logs:,} logs, {len(template_data):,} unique templates...")
    
    if missing_node_count > 0:
        print(f"Warning: {missing_node_count} logs had missing node_id")
    
    print(f"\nExtracted {len(template_data)} unique templates from {total_logs:,} logs")
    
    # Prepare templates list (from accumulated metadata)
    print("\nPreparing templates output...")
    templates = []
    for template_id, data in template_data.items():
        # Reconstruct template from first example
        template_text = extract_template(data["example_message"])
        
        templates.append({
            "template_id": template_id,
            "template": template_text,
            "example_message": data["example_message"],
            "count": data["count"]
        })
    
    # Sort by count (most common first) for determinism
    templates.sort(key=lambda x: (-x["count"], x["template_id"]))
    
    # Statistics
    print(f"\nTemplate Statistics:")
    print(f"  Total logs: {total_logs:,}")
    print(f"  Unique templates: {len(templates):,}")
    print(f"  Compression ratio: {total_logs/len(templates):.1f}x")
    print(f"  Most common template: {templates[0]['count']:,} occurrences")
    print(f"  Least common template: {templates[-1]['count']:,} occurrences")
    
    # Show top 5 templates
    print(f"\nTop 5 Templates:")
    for i, tmpl in enumerate(templates[:5], 1):
        print(f"  {i}. [{tmpl['count']:,} logs] {tmpl['template'][:80]}...")
    
    # Save templates
    print(f"\nSaving templates to {templates_output}...")
    with open(templates_output, 'w', encoding='utf-8') as f:
        for template in templates:
            f.write(json.dumps(template) + '\n')
    
    print(f"Saved {len(templates):,} templates")
    print(f"Template map saved to {template_map_output} ({total_logs:,} mappings)")
    
    print("\n" + "="*80)
    print(f"Template extraction complete!")
    print(f"Embedding cost reduced by {total_logs/len(templates):.1f}x")
    print(f"Peak memory: ~{len(template_data) * 0.001:.1f} MB (template metadata only)")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract BGL log templates")
    parser.add_argument("--input", default="data/processed/bgl_sampled_2000.json",
                       help="Input parsed logs JSON or JSONL file")
    parser.add_argument("--templates-output", default="data/processed/bgl/templates.jsonl",
                       help="Output templates JSONL file")
    parser.add_argument("--template-map-output", default="data/processed/bgl/template_map.jsonl",
                       help="Output template map JSONL file")
    parser.add_argument("--node-field", default="node_id",
                       help="Field name for node ID (default: node_id)")
    
    args = parser.parse_args()
    
    extract_templates(
        args.input,
        args.templates_output,
        args.template_map_output,
        args.node_field
    )
