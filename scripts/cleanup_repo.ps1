#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean up generated artifacts and move legacy files to archive.

.DESCRIPTION
    This script removes generated data, results, and logs from the LAR-RAG repository
    while preserving core code and documentation. It can optionally archive legacy files.

.PARAMETER ArchiveLegacy
    Move legacy files to docs/_archive instead of deleting them.

.PARAMETER DryRun
    Show what would be deleted without actually deleting anything.

.EXAMPLE
    .\scripts\cleanup_repo.ps1
    Clean up all generated files.

.EXAMPLE
    .\scripts\cleanup_repo.ps1 -ArchiveLegacy -DryRun
    Preview what would be archived without making changes.
#>

param(
    [switch]$ArchiveLegacy,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Danger { Write-Host $args -ForegroundColor Red }

Write-Info "========================================="
Write-Info "LAR-RAG Repository Cleanup Script"
Write-Info "========================================="

if ($DryRun) {
    Write-Warning "[DRY RUN MODE - No files will be deleted]"
}

# Get repository root
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Track what we delete
$DeletedFiles = @()
$DeletedDirs = @()
$ArchivedFiles = @()

# Function to safely remove item
function Remove-SafeItem {
    param(
        [string]$Path,
        [string]$Type = "File"
    )
    
    if (Test-Path $Path) {
        $relativePath = Resolve-Path -Relative $Path
        
        if ($DryRun) {
            Write-Warning "Would delete: $relativePath"
        } else {
            Remove-Item -Path $Path -Recurse -Force -ErrorAction SilentlyContinue
            Write-Success "Deleted: $relativePath"
        }
        
        if ($Type -eq "Directory") {
            $script:DeletedDirs += $relativePath
        } else {
            $script:DeletedFiles += $relativePath
        }
        
        return $true
    }
    return $false
}

# Function to archive file
function Move-ToArchive {
    param(
        [string]$Path
    )
    
    if (Test-Path $Path) {
        $relativePath = Resolve-Path -Relative $Path
        $archiveDir = Join-Path $RepoRoot "docs\_archive"
        
        if (-not (Test-Path $archiveDir) -and -not $DryRun) {
            New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
        }
        
        $fileName = Split-Path -Leaf $Path
        $archivePath = Join-Path $archiveDir $fileName
        
        if ($DryRun) {
            Write-Warning "Would archive: $relativePath -> docs\_archive\$fileName"
        } else {
            Move-Item -Path $Path -Destination $archivePath -Force
            Write-Success "Archived: $relativePath -> docs\_archive\$fileName"
        }
        
        $script:ArchivedFiles += $relativePath
        return $true
    }
    return $false
}

Write-Info "`n[1/5] Cleaning data directories..."

# Remove processed data (keep directory structure with .gitkeep)
$processedBglPath = Join-Path $RepoRoot "data\processed\bgl"
if (Remove-SafeItem $processedBglPath "Directory") {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path (Join-Path $RepoRoot "data\processed") -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $RepoRoot "data\processed\.gitkeep") -Force | Out-Null
    }
}

# Remove processed JSON files
Remove-SafeItem (Join-Path $RepoRoot "data\processed\bgl_sampled_2000.json")

# Remove raw BGL log (keep directory)
Remove-SafeItem (Join-Path $RepoRoot "data\raw\BGL\BGL.log")

# Remove topology files (these are generated)
Remove-SafeItem (Join-Path $RepoRoot "data\topology\hdfs_graph.json")

# Remove evaluation ground truth (regeneratable)
Remove-SafeItem (Join-Path $RepoRoot "data\evaluation\ground_truth.json")
Remove-SafeItem (Join-Path $RepoRoot "data\evaluation\hdfs_ground_truth.jsonl")

Write-Info "`n[2/5] Cleaning results and logs..."

# Remove result files but keep the analyze script
$resultsToRemove = @(
    "results\bgl_test_run_stats.json",
    "results\failed_batches.jsonl",
    "results\LAR-RAG_Baseline_and_Ablation_Study_20251217_140753.json",
    "results\LAR-RAG_Baseline_and_Ablation_Study_20251217_140753_metrics.csv",
    "results\LAR-RAG_Baseline_and_Ablation_Study_20251217_140753_statistical_tests.json"
)

foreach ($file in $resultsToRemove) {
    Remove-SafeItem (Join-Path $RepoRoot $file)
}

# Remove logs-raw
$logsRawPath = Join-Path $RepoRoot "logs-raw"
Remove-SafeItem $logsRawPath "Directory"

Write-Info "`n[3/5] Cleaning infrastructure state..."

# Remove Terraform state files (should not be in git)
Remove-SafeItem (Join-Path $RepoRoot "infra\terraform.tfstate")
Remove-SafeItem (Join-Path $RepoRoot "infra\terraform.tfstate.backup")
$terraformDir = Join-Path $RepoRoot "infra\.terraform"
Remove-SafeItem $terraformDir "Directory"

Write-Info "`n[4/5] Cleaning Python cache..."

# Remove __pycache__ directories
Get-ChildItem -Path $RepoRoot -Filter "__pycache__" -Recurse -Directory | ForEach-Object {
    Remove-SafeItem $_.FullName "Directory"
}

Write-Info "`n[5/5] Handling legacy files..."

if ($ArchiveLegacy) {
    # Archive implementation prompts
    $promptsToArchive = @(
        "docs\prompt.txt",
        "docs\PROMPT_BGL_EMBED_OPTIMISE.md",
        "docs\PROMPT_LAR_RAG_NEXT_STEPS.md",
        "docs\PROMPT_LAR_RAG_PAPER_READINESS_FIX.md",
        "docs\PROMPT_PATCH_TEMPLATE_AND_EMBED.md",
        "docs\PROMPT_REFACTOR_PARSE_BGL_STREAMING.md"
    )
    
    foreach ($file in $promptsToArchive) {
        Move-ToArchive (Join-Path $RepoRoot $file)
    }
    
    # Archive old ground truth builder
    Move-ToArchive (Join-Path $RepoRoot "src\evaluation\build_ground_truth_old.py")
} else {
    Write-Warning "Legacy files NOT archived. Use -ArchiveLegacy to move them to docs\_archive\"
}

# Summary
Write-Info "`n========================================="
Write-Info "Cleanup Summary"
Write-Info "========================================="

if ($DryRun) {
    Write-Warning "`n[DRY RUN - Nothing was actually deleted]`n"
}

Write-Info "Deleted files: $($DeletedFiles.Count)"
if ($DeletedFiles.Count -gt 0) {
    $DeletedFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
}

Write-Info "`nDeleted directories: $($DeletedDirs.Count)"
if ($DeletedDirs.Count -gt 0) {
    $DeletedDirs | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
}

if ($ArchiveLegacy) {
    Write-Info "`nArchived files: $($ArchivedFiles.Count)"
    if ($ArchivedFiles.Count -gt 0) {
        $ArchivedFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    }
}

Write-Info "`n========================================="
Write-Success "Cleanup complete!"
Write-Info "========================================="

if (-not $DryRun) {
    Write-Info "`nNext steps:"
    Write-Info "1. Review git status: git status"
    Write-Info "2. Stage deletions: git add -u"
    Write-Info "3. Commit: git commit -m 'Clean up generated artifacts'"
    Write-Info "`nTo regenerate data, run: .\scripts\reproduce_all.ps1"
}
