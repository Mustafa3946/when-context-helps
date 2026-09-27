# Download BGL Dataset
# Downloads the BGL log dataset from loghub Zenodo repository

Write-Host "="*80
Write-Host "Downloading BGL Dataset from Loghub"
Write-Host "="*80
Write-Host ""

$downloadUrl = "https://zenodo.org/records/3227177/files/BGL.tar.gz?download=1"
$outputFile = "BGL.tar.gz"
$extractPath = "data\raw\BGL"

# Download
Write-Host "Downloading BGL.tar.gz (~700MB)..."
Write-Host "This may take several minutes..."
Write-Host ""

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $outputFile -Verbose
    Write-Host "✓ Download complete!" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "Extracting BGL.tar.gz..."
try {
    # Ensure extract directory exists
    if (-not (Test-Path $extractPath)) {
        New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
    }
    
    # Extract using tar (available in Windows 10+)
    tar -xzf $outputFile -C $extractPath
    Write-Host "✓ Extraction complete!" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host "✗ Extraction failed: $_" -ForegroundColor Red
    Write-Host "Please extract manually to: $extractPath" -ForegroundColor Yellow
    exit 1
}

# Verify
$bglLogPath = Join-Path $extractPath "BGL.log"
if (Test-Path $bglLogPath) {
    $fileSize = (Get-Item $bglLogPath).Length / 1MB
    Write-Host "="*80
    Write-Host "✓ BGL Dataset Ready!" -ForegroundColor Green
    Write-Host "="*80
    Write-Host "Location: $bglLogPath"
    Write-Host "Size: $([math]::Round($fileSize, 2)) MB"
    Write-Host ""
    Write-Host "You can now run the pipeline:"
    Write-Host "  python scripts/run_bgl_pipeline_to_blob.py" -ForegroundColor Cyan
    Write-Host "="*80
    
    # Cleanup
    Remove-Item $outputFile -Force
    Write-Host "Cleaned up temporary files."
}
else {
    Write-Host "✗ Warning: BGL.log not found after extraction" -ForegroundColor Yellow
    Write-Host "Expected location: $bglLogPath"
    Write-Host "Please check the extracted files manually."
}
