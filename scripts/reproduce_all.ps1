# ============================================================================
# LAR-RAG Reproducibility Script
# ============================================================================
# This script reproduces all experiments from the LAR-RAG paper.
# 
# Prerequisites:
# 1. Python 3.10+ with virtual environment configured
# 2. Azure AI Search and Azure OpenAI credentials in .env file
# 3. Internet connection for downloading datasets
#
# Usage:
#   .\scripts\reproduce_all.ps1
#
# Estimated time: 2-3 hours (depends on Azure TPM limits)
# Estimated cost: $5-10 USD (Azure OpenAI embeddings)
# ============================================================================

param(
    [switch]$SkipDownload,     # Skip dataset download if already present
    [switch]$SkipEmbed,        # Skip embedding if already done
    [switch]$SkipGroundTruth,  # Skip ground truth generation if exists
    [int]$NumIncidents = 30    # Number of incidents to generate (default: 30)
)

# Stop on errors
$ErrorActionPreference = "Stop"

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "LAR-RAG Reproducibility Script" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 0. Setup and Validation
# ============================================================================
Write-Host "[Step 0] Validating environment..." -ForegroundColor Yellow

# Check Python environment
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

$pythonCmd = ".\.venv\Scripts\python.exe"

# Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found" -ForegroundColor Red
    Write-Host "Please create .env with Azure credentials (see .env.example)" -ForegroundColor Red
    exit 1
}

# Verify required packages
Write-Host "Checking Python packages..." -ForegroundColor Gray
$packages = & $pythonCmd -m pip list
if (-not ($packages -match "azure-search-documents")) {
    Write-Host "ERROR: azure-search-documents not installed" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Environment validated" -ForegroundColor Green
Write-Host ""

# ============================================================================
# 1. Download Datasets
# ============================================================================
if (-not $SkipDownload) {
    Write-Host "[Step 1] Downloading datasets from LogHub..." -ForegroundColor Yellow
    
    $dataDir = "data\raw"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    }
    
    # HDFS Dataset
    $hdfsUrl = "https://zenodo.org/records/8196385/files/HDFS_v1.zip"
    $hdfsZip = "$dataDir\HDFS_v1.zip"
    $hdfsExtract = "$dataDir\HDFS"
    
    if (-not (Test-Path $hdfsExtract)) {
        Write-Host "Downloading HDFS dataset (4.6 GB)..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $hdfsUrl -OutFile $hdfsZip
        
        Write-Host "Extracting HDFS dataset..." -ForegroundColor Gray
        Expand-Archive -Path $hdfsZip -DestinationPath $hdfsExtract -Force
        Remove-Item $hdfsZip
        
        Write-Host "✓ HDFS dataset downloaded" -ForegroundColor Green
    } else {
        Write-Host "✓ HDFS dataset already exists" -ForegroundColor Green
    }
    
    # BGL Dataset (optional for Task 9)
    if (Test-Path "config\enable_bgl.txt") {
        $bglUrl = "https://zenodo.org/records/8196385/files/BGL.zip"
        $bglZip = "$dataDir\BGL.zip"
        $bglExtract = "$dataDir\BGL"
        
        if (-not (Test-Path $bglExtract)) {
            Write-Host "Downloading BGL dataset (700 MB)..." -ForegroundColor Gray
            Invoke-WebRequest -Uri $bglUrl -OutFile $bglZip
            
            Write-Host "Extracting BGL dataset..." -ForegroundColor Gray
            Expand-Archive -Path $bglZip -DestinationPath $bglExtract -Force
            Remove-Item $bglZip
            
            Write-Host "✓ BGL dataset downloaded" -ForegroundColor Green
        } else {
            Write-Host "✓ BGL dataset already exists" -ForegroundColor Green
        }
    }
    
    Write-Host ""
} else {
    Write-Host "[Step 1] Skipping dataset download (--SkipDownload)" -ForegroundColor Gray
    Write-Host ""
}

# ============================================================================
# 2. Parse and Sample Logs
# ============================================================================
Write-Host "[Step 2] Parsing and sampling logs..." -ForegroundColor Yellow

# HDFS parsing (sample 2,000 logs)
$parsedHdfs = "data\processed\hdfs_sampled_2000.json"
if (-not (Test-Path $parsedHdfs)) {
    Write-Host "Parsing HDFS logs (sampling 2,000 from 11M)..." -ForegroundColor Gray
    & $pythonCmd src\data\parse_hdfs.py `
        --input data\raw\HDFS\HDFS_v1.log `
        --output $parsedHdfs `
        --sample 2000 `
        --random-seed 42
    
    Write-Host "✓ HDFS logs parsed and sampled" -ForegroundColor Green
} else {
    Write-Host "✓ HDFS logs already parsed" -ForegroundColor Green
}

Write-Host ""

# ============================================================================
# 3. Embed Logs with Azure OpenAI
# ============================================================================
if (-not $SkipEmbed) {
    Write-Host "[Step 3] Embedding logs with Azure OpenAI..." -ForegroundColor Yellow
    Write-Host "Estimated time: 4 minutes (2,000 logs @ 120K TPM)" -ForegroundColor Gray
    Write-Host "Estimated cost: ~$0.40 USD" -ForegroundColor Gray
    
    $embeddedHdfs = "data\processed\hdfs_embedded_2000.json"
    if (-not (Test-Path $embeddedHdfs)) {
        & $pythonCmd src\data\embed_logs.py `
            --input $parsedHdfs `
            --output $embeddedHdfs `
            --batch-size 100
        
        Write-Host "✓ HDFS logs embedded" -ForegroundColor Green
    } else {
        Write-Host "✓ HDFS logs already embedded" -ForegroundColor Green
    }
    
    Write-Host ""
} else {
    Write-Host "[Step 3] Skipping embedding (--SkipEmbed)" -ForegroundColor Gray
    Write-Host ""
}

# ============================================================================
# 4. Index Logs in Azure AI Search
# ============================================================================
Write-Host "[Step 4] Indexing logs in Azure AI Search..." -ForegroundColor Yellow

$indexName = "hdfs-logs"
& $pythonCmd src\data\index_logs.py `
    --input data\processed\hdfs_embedded_2000.json `
    --index-name $indexName `
    --recreate-index

Write-Host "✓ Logs indexed in Azure AI Search" -ForegroundColor Green
Write-Host ""

# ============================================================================
# 5. Generate Ground Truth
# ============================================================================
if (-not $SkipGroundTruth) {
    Write-Host "[Step 5] Generating ground truth with content-based protocol..." -ForegroundColor Yellow
    
    $groundTruthFile = "data\evaluation\hdfs_ground_truth.jsonl"
    if (-not (Test-Path $groundTruthFile)) {
        & $pythonCmd src\evaluation\ground_truth.py `
            --min-relevant 3 `
            --target-incidents $NumIncidents `
            --output $groundTruthFile
        
        Write-Host "✓ Ground truth generated ($NumIncidents incidents)" -ForegroundColor Green
    } else {
        Write-Host "✓ Ground truth already exists" -ForegroundColor Green
    }
    
    Write-Host ""
} else {
    Write-Host "[Step 5] Skipping ground truth generation (--SkipGroundTruth)" -ForegroundColor Gray
    Write-Host ""
}

# ============================================================================
# 6. Run Experiments
# ============================================================================
Write-Host "[Step 6] Running LAR-RAG experiments..." -ForegroundColor Yellow
Write-Host "Models: BM25, Dense, Hybrid, Temporal, Topology, LAR-RAG (Full)" -ForegroundColor Gray

# Create experiment config if doesn't exist
$configFile = "experiments\reproducibility_config.yaml"
if (-not (Test-Path $configFile)) {
    Write-Host "Creating experiment configuration..." -ForegroundColor Gray
    & $pythonCmd -c @"
from src.evaluation.experiment_runner import create_default_config
create_default_config('$configFile')
"@
}

# Run experiments
& $pythonCmd src\evaluation\experiment_runner.py `
    --config $configFile `
    --output-dir results

Write-Host "✓ Experiments completed" -ForegroundColor Green
Write-Host ""

# ============================================================================
# 7. Run Statistical Tests
# ============================================================================
Write-Host "[Step 7] Running statistical tests..." -ForegroundColor Yellow

# Find latest results directory
$latestResults = Get-ChildItem -Path "results" -Directory | 
    Sort-Object CreationTime -Descending | 
    Select-Object -First 1

if ($latestResults) {
    Write-Host "Analyzing results from: $($latestResults.Name)" -ForegroundColor Gray
    
    # Compare LAR-RAG Full vs Hybrid baseline
    $hybridPath = "$($latestResults.FullName)\per_query_Hybrid_BM25Dense.jsonl"
    $larragPath = "$($latestResults.FullName)\per_query_LAR-RAG_Full.jsonl"
    
    if ((Test-Path $hybridPath) -and (Test-Path $larragPath)) {
        & $pythonCmd -c @"
from src.evaluation.stats import PairedStatisticalTests

print('\n' + '='*80)
print('PRIMARY COMPARISON: LAR-RAG (Full) vs Hybrid Baseline')
print('='*80 + '\n')

comparison = PairedStatisticalTests.compare_systems(
    baseline_path='$hybridPath',
    model_path='$larragPath',
    metric='reciprocal_rank',
    verbose=True
)

# Also compare other metrics
print('\n' + '='*80)
print('Average Precision Comparison')
print('='*80 + '\n')

PairedStatisticalTests.compare_systems(
    baseline_path='$hybridPath',
    model_path='$larragPath',
    metric='average_precision',
    verbose=True
)
"@
        
        Write-Host "✓ Statistical tests completed" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Could not find per-query result files" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING: No results directory found" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# 8. Generate Paper Tables and Figures
# ============================================================================
Write-Host "[Step 8] Generating tables and figures for paper..." -ForegroundColor Yellow

if ($latestResults) {
    # Create LaTeX table
    & $pythonCmd scripts\generate_paper_tables.py `
        --results-dir $latestResults.FullName `
        --output-dir paper\tables
    
    # Create plots
    & $pythonCmd scripts\generate_paper_plots.py `
        --results-dir $latestResults.FullName `
        --output-dir paper\figures
    
    Write-Host "✓ Tables and figures generated" -ForegroundColor Green
} else {
    Write-Host "WARNING: No results to generate tables from" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# 9. Validation Summary
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "REPRODUCIBILITY VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check key results
if ($latestResults) {
    $summaryFile = "$($latestResults.FullName)\summary.csv"
    if (Test-Path $summaryFile) {
        Write-Host "Results Summary:" -ForegroundColor Green
        Get-Content $summaryFile | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Gray
        }
        Write-Host ""
    }
}

# Expected results (from paper)
Write-Host "Expected Results (from paper):" -ForegroundColor Yellow
Write-Host "  Hybrid Baseline:  MRR = 0.0511,  mAP = 0.0403" -ForegroundColor Gray
Write-Host "  LAR-RAG (Full):   MRR = 0.1279,  mAP = 0.1029" -ForegroundColor Gray
Write-Host "  Improvement:      +150.3% MRR,  +155.3% mAP" -ForegroundColor Gray
Write-Host "  Statistical Test: p < 0.001 (Wilcoxon)" -ForegroundColor Gray
Write-Host ""

Write-Host "If your results differ significantly:" -ForegroundColor Yellow
Write-Host "  1. Check random seed is fixed (42)" -ForegroundColor Gray
Write-Host "  2. Verify same dataset sampling (2,000 logs)" -ForegroundColor Gray
Write-Host "  3. Check Azure OpenAI model version (text-embedding-ada-002)" -ForegroundColor Gray
Write-Host "  4. Verify hyperparameters match config file" -ForegroundColor Gray
Write-Host ""

# List output files
Write-Host "Output Files:" -ForegroundColor Yellow
if ($latestResults) {
    Write-Host "  Results:  $($latestResults.FullName)" -ForegroundColor Gray
}
Write-Host "  Tables:   paper\tables\" -ForegroundColor Gray
Write-Host "  Figures:  paper\figures\" -ForegroundColor Gray
Write-Host "  Logs:     logs\reproducibility_$(Get-Date -Format 'yyyyMMdd_HHmmss').log" -ForegroundColor Gray
Write-Host ""

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "REPRODUCTION COMPLETE" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Compare results with expected values above" -ForegroundColor Gray
Write-Host "  2. Review statistical test output (should show p < 0.001)" -ForegroundColor Gray
Write-Host "  3. Check generated tables in paper/tables/" -ForegroundColor Gray
Write-Host "  4. Verify plots in paper/figures/" -ForegroundColor Gray
Write-Host "  5. See docs/REPRODUCTION.md for troubleshooting" -ForegroundColor Gray
Write-Host ""

# Return to original directory
Write-Host "Script completed successfully!" -ForegroundColor Green
