
# SAMPLE COMMAND
# .\run_inference.ps1 -ModelPath "pretrained/states_emoji_celebahq_5000.pth"

param(
    [int]$ModelIndex = -1,
    [string]$ModelPath = ""
)

# === CONFIGURATION (Easy to change) ===
$testBaseDir = "examples/emoji/open_test"  # Change this for different test sets

# === MODEL SELECTION ===
$models = Get-ChildItem -Path "pretrained" -Filter "*.pth" | Sort-Object Name

if ($models.Count -eq 0) {
    Write-Host "No .pth models found in pretrained/ directory!" -ForegroundColor Red
    exit 1
}

# Determine which model to use
if ($ModelPath -ne "") {
    if (-not (Test-Path $ModelPath)) {
        Write-Host "Model file not found: $ModelPath" -ForegroundColor Red
        exit 1
    }
    $selectedModel = $ModelPath
} elseif ($ModelIndex -ge 0) {
    if ($ModelIndex -ge $models.Count) {
        Write-Host "Invalid model index: $ModelIndex (max: $($models.Count - 1))" -ForegroundColor Red
        exit 1
    }
    $selectedModel = $models[$ModelIndex].FullName
} else {
    Write-Host "=== Available Models ===" -ForegroundColor Cyan
    for ($i = 0; $i -lt $models.Count; $i++) {
        Write-Host "  [$i] $($models[$i].Name)"
    }
    Write-Host ""

    do {
        $selection = Read-Host "Select model index (0-$($models.Count - 1))"
        $selectionInt = [int]$selection
    } while ($selectionInt -lt 0 -or $selectionInt -ge $models.Count)

    $selectedModel = $models[$selectionInt].FullName
}

$modelName = [System.IO.Path]::GetFileNameWithoutExtension($selectedModel)
Write-Host ""
Write-Host "Selected model: $modelName" -ForegroundColor Green
Write-Host "Test directory: $testBaseDir" -ForegroundColor Green
Write-Host ""

# === SETUP ===
$outputDir = "predictions_$modelName"
$metricsFile = "$outputDir/evaluation_metrics.txt"

# Create output directory
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Get list of erased images
$erasedImages = Get-ChildItem -Path "$testBaseDir/erased" -Filter "*.png"
$totalImages = $erasedImages.Count

if ($totalImages -eq 0) {
    Write-Host "No images found in $testBaseDir/erased/" -ForegroundColor Red
    exit 1
}

# === INFERENCE LOOP ===
Write-Host "=== Running Inference ===" -ForegroundColor Cyan
$current = 0

foreach ($erased in $erasedImages) {
    $current++

    # Derive paths
    $baseName = $erased.BaseName -replace '_erased$', ''
    $maskPath = "$testBaseDir/masks/${baseName}_mask.png"
    $outPath = "$outputDir/${baseName}_out_${modelName}.png"

    Write-Host "[$current/$totalImages] Processing: $($erased.Name)" -ForegroundColor Yellow

    # Run inference
    python test.py --image $erased.FullName --mask $maskPath --out $outPath --checkpoint $selectedModel
}

Write-Host ""
Write-Host "Inference completed for $totalImages images." -ForegroundColor Green
Write-Host ""

# === EVALUATION LOOP ===
Write-Host "=== Running Evaluation ===" -ForegroundColor Cyan
$results = @()
$current = 0

foreach ($erased in $erasedImages) {
    $current++

    $baseName = $erased.BaseName -replace '_erased$', ''
    $predPath = "$outputDir/${baseName}_out_${modelName}.png"
    $origPath = "$testBaseDir/orig/${baseName}.png"
    $maskPath = "$testBaseDir/masks/${baseName}_mask.png"

    Write-Host "[$current/$totalImages] Evaluating: $baseName" -ForegroundColor Yellow

    # Compute L1
    $l1Output = python compute_l1.py --pred $predPath --orig $origPath --mask $maskPath

    # Parse output
    $l1Full = 0.0
    $l1Masked = 0.0
    foreach ($line in $l1Output) {
        if ($line -match "^full:(.+)$") {
            $l1Full = [double]$Matches[1]
        } elseif ($line -match "^masked:(.+)$") {
            $l1Masked = [double]$Matches[1]
        }
    }

    $results += [PSCustomObject]@{
        Name = $baseName
        L1Full = $l1Full
        L1Masked = $l1Masked
    }
}

Write-Host ""
Write-Host "Evaluation completed." -ForegroundColor Green
Write-Host ""

# === COMPUTE STATISTICS ===
$l1FullValues = $results | ForEach-Object { $_.L1Full }
$l1MaskedValues = $results | ForEach-Object { $_.L1Masked }

$meanFull = ($l1FullValues | Measure-Object -Average).Average
$minFull = ($l1FullValues | Measure-Object -Minimum).Minimum
$maxFull = ($l1FullValues | Measure-Object -Maximum).Maximum

$meanMasked = ($l1MaskedValues | Measure-Object -Average).Average
$minMasked = ($l1MaskedValues | Measure-Object -Minimum).Minimum
$maxMasked = ($l1MaskedValues | Measure-Object -Maximum).Maximum

# Compute standard deviation
function Get-StdDev($values) {
    $mean = ($values | Measure-Object -Average).Average
    $sumSquares = 0
    foreach ($v in $values) {
        $sumSquares += ($v - $mean) * ($v - $mean)
    }
    return [Math]::Sqrt($sumSquares / $values.Count)
}

$stdFull = Get-StdDev $l1FullValues
$stdMasked = Get-StdDev $l1MaskedValues

# === WRITE METRICS FILE ===
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$metricsContent = @"
===========================================
Evaluation Results
===========================================
Model: $modelName
Test Directory: $testBaseDir
Date: $timestamp

Per-Image Results:
------------------------------------------
Image                              L1 Full        L1 Masked
"@

foreach ($r in $results | Sort-Object Name) {
    $nameFormatted = $r.Name.PadRight(30)
    $fullFormatted = "{0,12:F6}" -f $r.L1Full
    $maskedFormatted = "{0,12:F6}" -f $r.L1Masked
    $metricsContent += "`n$nameFormatted $fullFormatted $maskedFormatted"
}

$metricsContent += @"

------------------------------------------

Summary Statistics:
------------------------------------------
                                   L1 Full       L1 Masked
Mean                        $("{0,12:F6}" -f $meanFull) $("{0,12:F6}" -f $meanMasked)
Min                         $("{0,12:F6}" -f $minFull) $("{0,12:F6}" -f $minMasked)
Max                         $("{0,12:F6}" -f $maxFull) $("{0,12:F6}" -f $maxMasked)
Std Dev                     $("{0,12:F6}" -f $stdFull) $("{0,12:F6}" -f $stdMasked)
Num Images                  $("{0,12}" -f $totalImages) $("{0,12}" -f $totalImages)
===========================================
"@

$metricsContent | Out-File -FilePath $metricsFile -Encoding UTF8

Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Mean L1 Full:    $("{0:F4}" -f $meanFull)"
Write-Host "Mean L1 Masked:  $("{0:F4}" -f $meanMasked)"
Write-Host ""
Write-Host "Predictions saved to: $outputDir/" -ForegroundColor Green
Write-Host "Metrics saved to: $metricsFile" -ForegroundColor Green
