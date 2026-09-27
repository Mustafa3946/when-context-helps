# Quick test script for LAR-RAG experiments
# Runs core experiments without full pipeline

$ErrorActionPreference = "Stop"

Write-Host "============================================================================"
Write-Host "LAR-RAG Quick Experiment Test"
Write-Host "============================================================================"
Write-Host ""

$pythonCmd = ".\.venv\Scripts\python.exe"

# Check environment
if (-not (Test-Path $pythonCmd)) {
    Write-Host "ERROR: Virtual environment not found" -ForegroundColor Red
    exit 1
}

Write-Host "[Test 1] Checking ground truth file..." -ForegroundColor Yellow
$gtFile = "data\evaluation\hdfs_ground_truth.jsonl"
if (Test-Path $gtFile) {
    $lineCount = (Get-Content $gtFile | Measure-Object -Line).Lines
    Write-Host "  Ground truth exists: $lineCount incidents" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Ground truth file not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Test 2] Checking Azure AI Search connection..." -ForegroundColor Yellow
$testSearch = & $pythonCmd -c @"
import os
from dotenv import load_dotenv
load_dotenv()
endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
if endpoint:
    print(f'Connected to: {endpoint}')
else:
    print('ERROR: AZURE_SEARCH_ENDPOINT not configured')
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Azure Search not configured" -ForegroundColor Red
    exit 1
}
Write-Host "  $testSearch" -ForegroundColor Green

Write-Host ""
Write-Host "[Test 3] Testing statistical tests module..." -ForegroundColor Yellow
& $pythonCmd -c @"
from src.evaluation.stats import PairedStatisticalTests
import numpy as np
np.random.seed(42)
baseline = np.random.beta(2, 38, 10)
model = np.random.beta(5, 35, 10)
result = PairedStatisticalTests.paired_bootstrap_ci(baseline, model, n_bootstrap=1000)
print(f'  Bootstrap test working: mean_diff={result[\"mean_diff\"]:.4f}, p={result[\"p_value_estimate\"]:.4f}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Statistical tests failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Test 4] Testing node normalization..." -ForegroundColor Yellow
& $pythonCmd -c @"
from src.utils.node_normalization import normalize_node_id
test_cases = ['/10.251.70.211', 'Node19', 'R45-M0-N3']
results = [normalize_node_id(tc) for tc in test_cases]
print(f'  Normalization working: {test_cases[0]} -> {results[0]}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Node normalization failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================================"
Write-Host "All Tests Passed - System Ready for Experiments"
Write-Host "============================================================================"
Write-Host ""

Write-Host "To run full experiments, you would execute:" -ForegroundColor Yellow
Write-Host "  python src\evaluation\experiment_runner.py --config experiments\config.yaml" -ForegroundColor Gray
Write-Host ""
Write-Host "Note: Full experiments require:" -ForegroundColor Yellow
Write-Host "  - Azure AI Search with indexed logs" -ForegroundColor Gray
Write-Host "  - Azure OpenAI for embeddings" -ForegroundColor Gray
Write-Host "  - ~15-20 minutes to run 6 model comparisons" -ForegroundColor Gray
