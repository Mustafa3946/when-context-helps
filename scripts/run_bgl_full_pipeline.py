"""
BGL Full-Scale Pipeline Orchestrator

End-to-end processing of 700MB BGL dataset:
1. Parse full dataset (streaming, JSONL output)
2. Extract templates (deduplicate logs)
3. Embed templates (cost-optimized)
4. Run sanity checks

Reproducible, resumable, no manual steps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os
import json
import yaml
import time
from datetime import datetime
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = "config/bgl_full_run.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def check_prerequisites(config: dict):
    """Verify all prerequisites before starting pipeline."""
    print("Checking prerequisites...")
    
    # Check raw data exists
    raw_file = Path(config['raw_bgl_file'])
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw BGL file not found: {raw_file}\n"
            f"Please download BGL.log and place it in {raw_file.parent}/"
        )
    
    size_mb = raw_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Raw BGL file exists: {size_mb:.1f} MB")
    
    # Check Azure credentials
    required_env_vars = [
        'AZURE_OPENAI_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
    ]
    
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        print(f"  Current environment variables:")
        for var in required_env_vars:
            val = os.getenv(var)
            print(f"    {var}: {'✓ Set' if val else '✗ Missing'}")
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}\n"
            f"Please set them in .env file"
        )
    
    print(f"  ✓ Azure credentials configured")
    
    # Check Python scripts exist
    required_scripts = [
        'src/data/parse_bgl.py',
        'src/preprocess/template_bgl.py',
        'src/embeddings/embed_templates_safe_v2.py'  # Updated to v2 (production)
    ]
    
    for script in required_scripts:
        if not Path(script).exists():
            raise FileNotFoundError(f"Required script not found: {script}")
    
    print(f"  ✓ All required scripts present")
    print()


def run_step(name: str, command: list, check_output: str = None):
    """
    Run a pipeline step and verify success.
    
    Args:
        name: Step name for logging
        command: Command to execute
        check_output: Optional path to verify output was created
    """
    print("="*80)
    print(f"STEP: {name}")
    print("="*80)
    print(f"Command: {' '.join(command)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print()
        print(f"✓ Completed in {elapsed:.1f}s")
        
        # Verify output if specified
        if check_output:
            output_path = Path(check_output)
            if not output_path.exists():
                raise FileNotFoundError(f"Expected output not created: {check_output}")
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Output: {check_output} ({size_mb:.1f} MB)")
        
        print()
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ FAILED with exit code {e.returncode}")
        print()
        raise


def run_parsing(config: dict):
    """Step 1: Parse full BGL dataset with streaming."""
    command = [
        sys.executable,
        'src/data/parse_bgl.py',
        '--input', config['raw_bgl_file'],
        '--output', config['parsed_output'],
        '--mode', config['parse_mode'],
        '--progress-every', str(config['progress_every'])
    ]
    
    if config.get('max_lines'):
        command.extend(['--max-lines', str(config['max_lines'])])
    
    run_step(
        "Parse BGL Logs (Streaming)",
        command,
        check_output=config['parsed_output']
    )
    
    # Load and validate parse stats
    stats_file = Path(config['parsed_output']).parent / 'parse_stats.json'
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Sanity checks
    min_records = config['sanity_checks']['min_parsed_records']
    min_size_mb = config['sanity_checks']['min_parsed_size_mb']
    
    if stats['records_written'] < min_records:
        print(f"⚠️  WARNING: Only {stats['records_written']:,} records written (expected ≥ {min_records:,})")
    
    output_size_mb = Path(config['parsed_output']).stat().st_size / (1024 * 1024)
    if output_size_mb < min_size_mb:
        print(f"⚠️  WARNING: Output only {output_size_mb:.1f} MB (expected ≥ {min_size_mb} MB)")
    
    return stats


def run_template_extraction(config: dict):
    """Step 2: Extract templates from parsed logs."""
    command = [
        sys.executable,
        'src/preprocess/template_bgl.py',
        '--input', config['parsed_output'],
        '--templates-output', config['templates_output'],
        '--template-map-output', config['template_map_output'],
        '--node-field', config['node_field']
    ]
    
    run_step(
        "Extract Templates (Deduplication)",
        command,
        check_output=config['templates_output']
    )
    
    # Count templates
    template_count = 0
    with open(config['templates_output'], 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                template_count += 1
    
    print(f"Total unique templates: {template_count:,}")
    
    # Sanity check
    min_templates = config['sanity_checks']['min_templates']
    if template_count < min_templates:
        print(f"⚠️  WARNING: Only {template_count:,} templates (expected ≥ {min_templates:,})")
    
    print()
    
    return template_count


def run_embedding(config: dict):
    """Step 3: Embed templates with Azure OpenAI (using v2 - production)."""
    emb_config = config['embedding']
    
    command = [
        sys.executable,
        'src/embeddings/embed_templates_safe_v2.py',  # Updated to v2
        '--templates', config['templates_output'],
        '--output', config['template_embeddings_output'],
        '--batch-size', str(emb_config['batch_size'])
        # Note: v2 removed --price-per-1m-tokens (unreliable)
    ]
    
    if emb_config['resume']:
        command.append('--resume')
    else:
        command.append('--no-resume')
    
    # v2 specific: add non-interactive flag for automation
    command.append('--non-interactive')
    
    # Add --force to override safety checks for large template counts
    command.append('--force')
    
    run_step(
        "Embed Templates (Azure OpenAI v2 - Production)",
        command,
        check_output=config['template_embeddings_output']
    )
    
    # Count embeddings from sharded Parquet output (v2 format)
    embedding_count = 0
    output_path = Path(config['template_embeddings_output'])
    
    if output_path.is_dir():
        # v2: Sharded output directory with part-*.parquet files
        try:
            import pyarrow.parquet as pq
            for shard_file in output_path.glob('part-*.parquet'):
                table = pq.read_table(shard_file)
                embedding_count += len(table)
        except ImportError:
            print("  ⚠️  Warning: pyarrow not available, cannot count embeddings")
            embedding_count = -1
    else:
        # Legacy: Single file (v1 or old format)
        if output_path.suffix == '.parquet':
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(output_path)
                embedding_count = len(table)
            except ImportError:
                print("  ⚠️  Warning: pyarrow not available")
                embedding_count = -1
        else:
            # JSONL fallback
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        embedding_count += 1
    
    if embedding_count >= 0:
        print(f"Total embeddings created: {embedding_count:,}")
    print()
    
    return embedding_count


def run_sanity_checks(config: dict, parse_stats: dict, template_count: int, embedding_count: int):
    """Step 4: Automated sanity checks."""
    print("="*80)
    print("STEP: Sanity Checks")
    print("="*80)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Count consistency
    checks_total += 1
    print("1. Count Consistency Check")
    print(f"   Parsed logs: {parse_stats['records_written']:,}")
    print(f"   Unique templates: {template_count:,}")
    print(f"   Embeddings: {embedding_count:,}")
    
    if embedding_count >= template_count * 0.95:  # Allow 5% failure tolerance
        print("   ✓ Counts are consistent")
        checks_passed += 1
    else:
        print(f"   ✗ Embedding count too low (expected ≥ {template_count * 0.95:.0f})")
    print()
    
    # Check 2: Embedding shape
    checks_total += 1
    print("2. Embedding Shape Check")
    
    output_path = Path(config['template_embeddings_output'])
    embeddings = []
    
    # Handle v2 sharded Parquet format (directory) or legacy JSONL (file)
    if output_path.is_dir():
        # v2: Read from first shard
        try:
            import pyarrow.parquet as pq
            first_shard = next(output_path.glob('part-*.parquet'), None)
            if first_shard:
                table = pq.read_table(first_shard)
                # Get first 10 embeddings
                for i in range(min(10, len(table))):
                    row = table.slice(i, 1).to_pydict()
                    embeddings.append(row['embedding'][0])
        except ImportError:
            print("   ⚠️  Warning: pyarrow not available, skipping shape check")
    else:
        # Legacy JSONL format
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                if line.strip():
                    record = json.loads(line)
                    embeddings.append(record['embedding'])
    
    if embeddings:
        dims = [len(emb) for emb in embeddings]
        all_same = all(d == dims[0] for d in dims)
        no_zeros = all(any(v != 0 for v in emb) for emb in embeddings)
        
        print(f"   Dimensions: {dims[0]} (expected 1536 for text-embedding-3-small)")
        print(f"   Consistent: {all_same}")
        print(f"   Non-zero: {no_zeros}")
        
        if all_same and no_zeros and dims[0] == 1536:
            print("   ✓ Embeddings have correct shape")
            checks_passed += 1
        else:
            print("   ✗ Embedding shape issues detected")
    else:
        print("   ⚠️  Could not load embeddings for shape check")
    print()
    
    # Check 3: Spot retrieval (simplified for large datasets)
    checks_total += 1
    print("3. Spot Retrieval Test")
    print("   Verifying embedding structure...")
    
    # For v2 sharded format, just verify we can read the data
    if output_path.is_dir():
        try:
            import pyarrow.parquet as pq
            shard_count = len(list(output_path.glob('part-*.parquet')))
            manifest_file = output_path / 'manifest.json'
            
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                print(f"   ✓ Found {shard_count} shards with manifest")
                print(f"   Model: {manifest.get('model', 'unknown')}")
                print(f"   Total embeddings: {manifest.get('total_successfully_embedded', 0):,}")
                checks_passed += 1
            else:
                print(f"   ⚠️  Found {shard_count} shards but no manifest")
        except Exception as e:
            print(f"   ✗ Error reading sharded output: {e}")
    else:
        # Legacy: Load sample embeddings
        template_embeddings = {}
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100
                    break
                if line.strip():
                    record = json.loads(line)
                    template_embeddings[record['template_id']] = {
                        'template': record['template'],
                        'embedding': record['embedding'],
                        'count': record['count']
                    }
        
        if template_embeddings:
            print(f"   ✓ Loaded {len(template_embeddings):,} sample embeddings")
            print(f"   Sample template IDs: {list(template_embeddings.keys())[:3]}")
            checks_passed += 1
        else:
            print("   ✗ No template embeddings loaded")
    
    print()
    
    # Summary
    print("="*80)
    print(f"Sanity Checks: {checks_passed}/{checks_total} passed")
    print("="*80)
    print()
    
    return checks_passed == checks_total


def save_run_statistics(config: dict, parse_stats: dict, template_count: int, embedding_count: int):
    """Save comprehensive run statistics."""
    
    output_path = Path(config['stats_output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate sizes
    parsed_size_mb = Path(config['parsed_output']).stat().st_size / (1024 * 1024)
    templates_size_mb = Path(config['templates_output']).stat().st_size / (1024 * 1024)
    
    # Handle v2 sharded output (directory) or legacy single file
    embeddings_path = Path(config['template_embeddings_output'])
    if embeddings_path.is_dir():
        # Sum all shard sizes
        embeddings_size_mb = sum(f.stat().st_size for f in embeddings_path.glob('part-*.parquet')) / (1024 * 1024)
    else:
        embeddings_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    
    # Compression ratio
    compression_ratio = parse_stats['records_written'] / template_count if template_count > 0 else 0
    
    # Estimate cost (rough)
    avg_tokens_per_template = 50  # Conservative estimate
    total_tokens = template_count * avg_tokens_per_template
    estimated_cost = (total_tokens / 1_000_000) * config['embedding']['price_per_1m_tokens']
    
    stats = {
        'pipeline_run': {
            'timestamp': datetime.now().isoformat(),
            'config': config['raw_bgl_file']
        },
        'parsing': {
            'lines_read': parse_stats['lines_read'],
            'records_written': parse_stats['records_written'],
            'unparseable_lines': parse_stats['unparseable_lines'],
            'missing_timestamp_pct': parse_stats['missing_timestamp_pct'],
            'missing_node_id_pct': parse_stats['missing_node_id_pct'],
            'output_size_mb': round(parsed_size_mb, 2)
        },
        'template_extraction': {
            'total_logs': parse_stats['records_written'],
            'unique_templates': template_count,
            'compression_ratio': round(compression_ratio, 2),
            'output_size_mb': round(templates_size_mb, 2)
        },
        'embedding': {
            'model': config['embedding']['model'],
            'templates_total': template_count,
            'embeddings_created': embedding_count,
            'estimated_tokens': total_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
            'output_size_mb': round(embeddings_size_mb, 2)
        },
        'cost_savings': {
            'raw_log_tokens': parse_stats['records_written'] * avg_tokens_per_template,
            'template_tokens': template_count * avg_tokens_per_template,
            'reduction_factor': round(compression_ratio, 2),
            'cost_saved_usd': round(estimated_cost * (compression_ratio - 1), 4)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("="*80)
    print("Final Statistics")
    print("="*80)
    print(f"Parsed logs: {stats['parsing']['records_written']:,}")
    print(f"Unique templates: {stats['template_extraction']['unique_templates']:,}")
    print(f"Compression ratio: {stats['template_extraction']['compression_ratio']}x")
    print(f"Embeddings: {stats['embedding']['embeddings_created']:,}")
    print(f"Estimated tokens: {stats['embedding']['estimated_tokens']:,}")
    print(f"Estimated cost: ${stats['embedding']['estimated_cost_usd']:.4f}")
    print(f"Cost savings: ${stats['cost_savings']['cost_saved_usd']:.4f} ({stats['cost_savings']['reduction_factor']}x reduction)")
    print()
    print(f"Statistics saved to: {output_path}")
    print("="*80)


def main():
    """Run full BGL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BGL Full-Scale Pipeline")
    parser.add_argument("--config", default="config/bgl_full_run.yaml",
                       help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    print()
    print("="*80)
    print("BGL Full-Scale Pipeline")
    print("Latency-Aware Retrieval for Root-Cause Analysis (LAR-RAG)")
    print("="*80)
    print()
    
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"Configuration loaded from: {args.config}")
        print(f"Mode: {config['parse_mode'].upper()}")
        print()
        
        # Check prerequisites
        check_prerequisites(config)
        
        # Run pipeline steps
        parse_stats = run_parsing(config)
        template_count = run_template_extraction(config)
        embedding_count = run_embedding(config)
        
        # Sanity checks
        all_checks_passed = run_sanity_checks(config, parse_stats, template_count, embedding_count)
        
        # Save statistics
        save_run_statistics(config, parse_stats, template_count, embedding_count)
        
        # Final status
        print()
        if all_checks_passed:
            print("✅ Pipeline completed successfully!")
        else:
            print("⚠️  Pipeline completed with warnings - review sanity checks")
        print()
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ PIPELINE FAILED")
        print("="*80)
        print(f"Error: {e}")
        print()
        raise


if __name__ == "__main__":
    main()
