"""
Paper Run Orchestrator - INFOCOM 2026

Executes the complete LAR-RAG evaluation pipeline:
1. Build occurrence store (if needed)
2. Build FAISS index (if needed)
3. Build ground truth (streaming)
4. Run experiments
5. Generate summary report

ACADEMIC ARTIFACT - REPRODUCIBLE PIPELINE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
import subprocess
from datetime import datetime
import yaml
import json

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete evaluation pipeline."""
    
    def __init__(self, config_path: str, force_rebuild: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to canonical v2 configuration
            force_rebuild: Force rebuild of all artifacts
        """
        self.config_path = Path(config_path)
        self.force_rebuild = force_rebuild
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("="*80)
        logger.info("PAPER RUN ORCHESTRATOR - INFOCOM 2026")
        logger.info("="*80)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Run name: {self.config['run']['name']}")
        logger.info(f"Seed: {self.config['run']['seed']}")
        logger.info(f"Force rebuild: {self.force_rebuild}")
        logger.info("="*80)
    
    def _run_command(self, cmd: list, description: str, check: bool = True):
        """Run a shell command."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP: {description}")
        logger.info(f"{'='*80}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=check)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
        else:
            logger.error(f"❌ {description} failed with code {result.returncode}")
            if check:
                raise RuntimeError(f"{description} failed")
        
        return result.returncode == 0
    
    def step1_build_occurrence_store(self):
        """Build occurrence store from template_map.jsonl."""
        db_path = Path(self.config['data']['occurrence_store']['db_path'])
        template_map_path = Path('data/processed/bgl/template_map.jsonl')
        
        if db_path.exists() and not self.force_rebuild:
            logger.info(f"Occurrence store already exists: {db_path}")
            logger.info("Skipping build (use --force-rebuild to regenerate)")
            return True
        
        cmd = [
            sys.executable, 'scripts/build_occurrence_store.py',
            '--template-map', str(template_map_path),
            '--output', str(db_path)
        ]
        
        return self._run_command(cmd, "Build occurrence store")
    
    def step2_build_faiss_index(self):
        """Build FAISS index from embeddings."""
        index_path = Path(self.config['retrieval']['faiss']['index_path'])
        embeddings_manifest = Path(self.config['data']['embeddings']['manifest_path'])
        
        if index_path.exists() and not self.force_rebuild:
            logger.info(f"FAISS index already exists: {index_path}")
            logger.info("Skipping build (use --force-rebuild to regenerate)")
            return True
        
        # Check if embeddings exist
        if not embeddings_manifest.exists():
            logger.error(f"Embeddings manifest not found: {embeddings_manifest}")
            logger.error("Run embedding generation first")
            return False
        
        embeddings_dir = embeddings_manifest.parent
        index_dir = index_path.parent
        
        cmd = [
            sys.executable, 'scripts/build_faiss_index.py',
            '--embeddings-dir', str(embeddings_dir),
            '--out', str(index_dir),
            '--normalize'
        ]
        
        return self._run_command(cmd, "Build FAISS index")
    
    def step2b_build_bm25_index(self):
        """Build BM25 index from templates."""
        index_dir = Path(self.config['retrieval']['bm25']['index_dir'])
        templates_path = Path(self.config['data']['templates_path'])
        
        if (index_dir / 'bm25_index.pkl').exists() and not self.force_rebuild:
            logger.info(f"BM25 index already exists: {index_dir}")
            logger.info("Skipping build (use --force-rebuild to regenerate)")
            return True
        
        cmd = [
            sys.executable, 'scripts/build_bm25_index.py',
            '--templates', str(templates_path),
            '--output', str(index_dir)
        ]
        
        return self._run_command(cmd, "Build BM25 index")
    
    def step3_build_ground_truth(self):
        """Build ground truth using streaming builder."""
        incidents_path = Path(self.config['ground_truth']['incidents_path'])
        qrels_path = Path(self.config['ground_truth']['qrels_path'])
        
        if incidents_path.exists() and qrels_path.exists() and not self.force_rebuild:
            logger.info(f"Ground truth already exists:")
            logger.info(f"  Incidents: {incidents_path}")
            logger.info(f"  Qrels: {qrels_path}")
            logger.info("Skipping build (use --force-rebuild to regenerate)")
            return True
        
        # Infer output base from incidents_path
        output_base = str(incidents_path).replace('_incidents.jsonl', '.jsonl')
        
        # Get config values with defaults
        gt_max = self.config['ground_truth'].get('max_incidents', 100)
        gt_min = self.config['ground_truth'].get('min_relevant', 3)
        
        cmd = [
            sys.executable, 'scripts/build_ground_truth_streaming.py',
            '--parsed-logs', 'data/processed/bgl/parsed.jsonl',
            '--template-map', 'data/processed/bgl/template_map.jsonl',
            '--output', output_base,
            '--seed', str(self.config['run']['seed']),
            '--max-incidents', str(gt_max),
            '--min-relevant', str(gt_min)
        ]
        
        return self._run_command(cmd, "Build ground truth (streaming)")
    
    def step4_run_experiments(self):
        """Run experiments using canonical v2 harness."""
        cmd = [
            sys.executable, '-m', 'src.evaluation.run_experiments_v2',
            '--config', str(self.config_path)
        ]
        
        return self._run_command(cmd, "Run experiments")
    
    def step5_generate_summary(self):
        """Generate summary report."""
        output_dir = Path(self.config['run']['output_dir'])
        results_json = output_dir / 'results.json'
        
        if not results_json.exists():
            logger.warning(f"Results file not found: {results_json}")
            return False
        
        with open(results_json, 'r') as f:
            results = json.load(f)
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*80)
        
        # Print table
        methods = list(results.keys())
        metrics = ['recall_at_10', 'mrr_at_10', 'ndcg_at_10', 'map', 'latency_p95_ms']
        
        # Header
        header = f"{'Method':<25} " + " ".join(f"{m:>12}" for m in metrics)
        logger.info(header)
        logger.info("-" * len(header))
        
        # Rows
        for method in methods:
            values = [results[method].get(m, 0.0) for m in metrics]
            row = f"{method:<25} " + " ".join(f"{v:>12.4f}" for v in values)
            logger.info(row)
        
        logger.info("="*80)
        logger.info(f"Full results saved to: {output_dir}")
        
        return True
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Occurrence store
            if not self.step1_build_occurrence_store():
                logger.error("Pipeline failed at step 1")
                return False
            
            # Step 2: FAISS index
            if not self.step2_build_faiss_index():
                logger.error("Pipeline failed at step 2")
                return False
            
            # Step 2b: BM25 index
            if not self.step2b_build_bm25_index():
                logger.error("Pipeline failed at step 2b")
                return False
            
            # Step 3: Ground truth
            if not self.step3_build_ground_truth():
                logger.error("Pipeline failed at step 3")
                return False
            
            # Step 4: Experiments
            if not self.step4_run_experiments():
                logger.error("Pipeline failed at step 4")
                return False
            
            # Step 5: Summary
            self.step5_generate_summary()
            
            # Success
            elapsed = datetime.now() - start_time
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {elapsed}")
            logger.info("="*80)
            
            return True
        
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete LAR-RAG evaluation pipeline (INFOCOM 2026)"
    )
    parser.add_argument(
        '--config',
        default='config/bgl_canonical_v2.yaml',
        help='Path to canonical v2 configuration'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of all artifacts'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = PipelineOrchestrator(args.config, args.force_rebuild)
    success = orchestrator.run_pipeline()
    
    sys.exit(0 if success else 1)
