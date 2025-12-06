
# SAMPLE COMMAND
# .\run_inference.ps1 -ModelPath "pretrained/states_emoji_no-ft-50000.pth"

param(
    [int]$ModelIndex = -1,
    [string]$ModelPath = ""
)

# === CONFIGURATION (Easy to change) ===
$testBaseDir = "examples/emoji/triplets_HTI"  # Change this for different test sets

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
$metricsFile = "$outputDir/evaluation_metrics_oti.txt"

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
    $l1Output = python ./scripts/compute_l1.py --pred $predPath --orig $origPath --mask $maskPath

    # Parse L1 output
    $l1Full = 0.0
    $l1Masked = 0.0
    foreach ($line in $l1Output) {
        if ($line -match "^full:(.+)$") {
            $l1Full = [double]$Matches[1]
        } elseif ($line -match "^masked:(.+)$") {
            $l1Masked = [double]$Matches[1]
        }
    }

    # Compute PSNR
    $psnrOutput = python ./scripts/compute_psnr.py --pred $predPath --orig $origPath --mask $maskPath

    # Parse PSNR output
    $psnrFull = 0.0
    $psnrMasked = 0.0
    foreach ($line in $psnrOutput) {
        if ($line -match "^full:(.+)$") {
            $psnrFull = [double]$Matches[1]
        } elseif ($line -match "^masked:(.+)$") {
            $psnrMasked = [double]$Matches[1]
        }
    }

    # Compute SSIM
    $ssimOutput = python ./scripts/compute_ssim.py --pred $predPath --orig $origPath --mask $maskPath

    # Parse SSIM output
    $ssimFull = 0.0
    $ssimMasked = 0.0
    foreach ($line in $ssimOutput) {
        if ($line -match "^full:(.+)$") {
            $ssimFull = [double]$Matches[1]
        } elseif ($line -match "^masked:(.+)$") {
            $ssimMasked = [double]$Matches[1]
        }
    }

    $results += [PSCustomObject]@{
        Name = $baseName
        L1Full = $l1Full
        L1Masked = $l1Masked
        PSNRFull = $psnrFull
        PSNRMasked = $psnrMasked
        SSIMFull = $ssimFull
        SSIMMasked = $ssimMasked
    }
}

Write-Host ""
Write-Host "Evaluation completed." -ForegroundColor Green
Write-Host ""

# === COMPUTE STATISTICS ===

# Compute standard deviation helper function
function Get-StdDev($values) {
    $mean = ($values | Measure-Object -Average).Average
    $sumSquares = 0
    foreach ($v in $values) {
        $sumSquares += ($v - $mean) * ($v - $mean)
    }
    return [Math]::Sqrt($sumSquares / $values.Count)
}

# L1 statistics
$l1FullValues = $results | ForEach-Object { $_.L1Full }
$l1MaskedValues = $results | ForEach-Object { $_.L1Masked }

$l1MeanFull = ($l1FullValues | Measure-Object -Average).Average
$l1MinFull = ($l1FullValues | Measure-Object -Minimum).Minimum
$l1MaxFull = ($l1FullValues | Measure-Object -Maximum).Maximum
$l1StdFull = Get-StdDev $l1FullValues

$l1MeanMasked = ($l1MaskedValues | Measure-Object -Average).Average
$l1MinMasked = ($l1MaskedValues | Measure-Object -Minimum).Minimum
$l1MaxMasked = ($l1MaskedValues | Measure-Object -Maximum).Maximum
$l1StdMasked = Get-StdDev $l1MaskedValues

# PSNR statistics
$psnrFullValues = $results | ForEach-Object { $_.PSNRFull }
$psnrMaskedValues = $results | ForEach-Object { $_.PSNRMasked }

$psnrMeanFull = ($psnrFullValues | Measure-Object -Average).Average
$psnrMinFull = ($psnrFullValues | Measure-Object -Minimum).Minimum
$psnrMaxFull = ($psnrFullValues | Measure-Object -Maximum).Maximum
$psnrStdFull = Get-StdDev $psnrFullValues

$psnrMeanMasked = ($psnrMaskedValues | Measure-Object -Average).Average
$psnrMinMasked = ($psnrMaskedValues | Measure-Object -Minimum).Minimum
$psnrMaxMasked = ($psnrMaskedValues | Measure-Object -Maximum).Maximum
$psnrStdMasked = Get-StdDev $psnrMaskedValues

# SSIM statistics
$ssimFullValues = $results | ForEach-Object { $_.SSIMFull }
$ssimMaskedValues = $results | ForEach-Object { $_.SSIMMasked }

$ssimMeanFull = ($ssimFullValues | Measure-Object -Average).Average
$ssimMinFull = ($ssimFullValues | Measure-Object -Minimum).Minimum
$ssimMaxFull = ($ssimFullValues | Measure-Object -Maximum).Maximum
$ssimStdFull = Get-StdDev $ssimFullValues

$ssimMeanMasked = ($ssimMaskedValues | Measure-Object -Average).Average
$ssimMinMasked = ($ssimMaskedValues | Measure-Object -Minimum).Minimum
$ssimMaxMasked = ($ssimMaskedValues | Measure-Object -Maximum).Maximum
$ssimStdMasked = Get-StdDev $ssimMaskedValues

# === WRITE METRICS FILE ===
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$metricsContent = @"
===============================================================================================================
Evaluation Results
===============================================================================================================
Model: $modelName
Test Directory: $testBaseDir
Date: $timestamp

Per-Image Results:
---------------------------------------------------------------------------------------------------------------
Image                              L1 Full    L1 Masked    PSNR Full  PSNR Masked    SSIM Full  SSIM Masked
"@

foreach ($r in $results | Sort-Object Name) {
    $nameFormatted = $r.Name.PadRight(30)
    $l1FullFmt = "{0,10:F4}" -f $r.L1Full
    $l1MaskedFmt = "{0,10:F4}" -f $r.L1Masked
    $psnrFullFmt = "{0,10:F4}" -f $r.PSNRFull
    $psnrMaskedFmt = "{0,10:F4}" -f $r.PSNRMasked
    $ssimFullFmt = "{0,10:F6}" -f $r.SSIMFull
    $ssimMaskedFmt = "{0,10:F6}" -f $r.SSIMMasked
    $metricsContent += "`n$nameFormatted $l1FullFmt $l1MaskedFmt $psnrFullFmt $psnrMaskedFmt $ssimFullFmt $ssimMaskedFmt"
}

$metricsContent += @"

---------------------------------------------------------------------------------------------------------------

Summary Statistics:
---------------------------------------------------------------------------------------------------------------
Metric              Full Image                              Masked Region
                    Mean        Min         Max         Std         Mean        Min         Max         Std
---------------------------------------------------------------------------------------------------------------
L1              $("{0,10:F4}" -f $l1MeanFull) $("{0,10:F4}" -f $l1MinFull) $("{0,10:F4}" -f $l1MaxFull) $("{0,10:F4}" -f $l1StdFull)    $("{0,10:F4}" -f $l1MeanMasked) $("{0,10:F4}" -f $l1MinMasked) $("{0,10:F4}" -f $l1MaxMasked) $("{0,10:F4}" -f $l1StdMasked)
PSNR            $("{0,10:F4}" -f $psnrMeanFull) $("{0,10:F4}" -f $psnrMinFull) $("{0,10:F4}" -f $psnrMaxFull) $("{0,10:F4}" -f $psnrStdFull)    $("{0,10:F4}" -f $psnrMeanMasked) $("{0,10:F4}" -f $psnrMinMasked) $("{0,10:F4}" -f $psnrMaxMasked) $("{0,10:F4}" -f $psnrStdMasked)
SSIM            $("{0,10:F6}" -f $ssimMeanFull) $("{0,10:F6}" -f $ssimMinFull) $("{0,10:F6}" -f $ssimMaxFull) $("{0,10:F6}" -f $ssimStdFull)    $("{0,10:F6}" -f $ssimMeanMasked) $("{0,10:F6}" -f $ssimMinMasked) $("{0,10:F6}" -f $ssimMaxMasked) $("{0,10:F6}" -f $ssimStdMasked)
---------------------------------------------------------------------------------------------------------------
Num Images: $totalImages
===============================================================================================================
"@

$metricsContent | Out-File -FilePath $metricsFile -Encoding UTF8

Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "                    Full Image       Masked Region"
Write-Host "Mean L1:        $("{0,12:F4}" -f $l1MeanFull) $("{0,12:F4}" -f $l1MeanMasked)"
Write-Host "Mean PSNR:      $("{0,12:F4}" -f $psnrMeanFull) $("{0,12:F4}" -f $psnrMeanMasked)"
Write-Host "Mean SSIM:      $("{0,12:F6}" -f $ssimMeanFull) $("{0,12:F6}" -f $ssimMeanMasked)"
Write-Host ""
Write-Host "Predictions saved to: $outputDir/" -ForegroundColor Green
Write-Host "Metrics saved to: $metricsFile" -ForegroundColor Green
