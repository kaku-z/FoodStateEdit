$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot

try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed with exit code $LASTEXITCODE"
    }

    $compileFiles = Get-ChildItem -LiteralPath $repoRoot -Recurse -Filter "*.py" -File |
        Where-Object { $_.FullName -notmatch "[\\/]__pycache__[\\/]" } |
        ForEach-Object FullName
    $compileBatchSize = 40
    for ($offset = 0; $offset -lt $compileFiles.Count; $offset += $compileBatchSize) {
        $lastIndex = [Math]::Min($offset + $compileBatchSize - 1, $compileFiles.Count - 1)
        $compileBatch = $compileFiles[$offset..$lastIndex]
        python -m py_compile @compileBatch
        if ($LASTEXITCODE -ne 0) {
            throw "Python compilation failed with exit code $LASTEXITCODE"
        }
    }

    python -c "import json, pathlib; excluded={'.git','node_modules'}; files=[p for p in pathlib.Path('.').rglob('*.json') if not excluded.intersection(p.parts)]; [json.loads(p.read_text(encoding='utf-8')) for p in files]; print(f'JSON parsed: {len(files)} files')"
    if ($LASTEXITCODE -ne 0) {
        throw "JSON validation failed with exit code $LASTEXITCODE"
    }

    $expectedHashes = @{
        "vendor_overrides\GeoEdit\geoedit\inference.py" = "D093AE7E2245884CCB7FC664B78A919EA5443088E77AA1A53C6A033F1E8C8BF1"
        "vendor_overrides\GeoEdit\diffsynth\pipelines\wan_video.py" = "50BB12685EF9CCB6B89D6600C27AD0CD2F12B5C67C5301FC51EA3F0EFF02BFB5"
        "vendor_overrides\GeoEdit\diffsynth\utils\data\__init__.py" = "FE056B4A675A345CF02D6C76D327E8434103CCF2A4066FF208CE41368771E360"
        "vendor_overrides\GeoEdit\tests\test_masks.py" = "FC6400E66C94A22E5048FBF7FC37058E6BAC099C8C2024086B7EA408E640F421"
        "results\day3_geoedit_overrides_v1\GeoEdit\geoedit\inference.py" = "F78488216536E70744E839C0294319F334D541ADC36F9E76136C14E7FFD328FA"
        "results\day3_geoedit_overrides_v1\GeoEdit\diffsynth\pipelines\wan_video.py" = "28A1B5E3939983139E2DCD4A7F19D085FCEC0017434793BAC163767B7AE29C00"
        "results\day3_geoedit_overrides_v1\GeoEdit\diffsynth\utils\data\__init__.py" = "FE056B4A675A345CF02D6C76D327E8434103CCF2A4066FF208CE41368771E360"
        "results\day3_geoedit_overrides_v1\GeoEdit\tests\test_masks.py" = "57176EA2595F236846E1D38A79F58E3D586A99E67CE7EF0DF4747AF768A982DA"
    }

    foreach ($relativePath in $expectedHashes.Keys) {
        $absolutePath = Join-Path $repoRoot $relativePath
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $absolutePath).Hash
        if ($actualHash -ne $expectedHashes[$relativePath]) {
            throw "Hash mismatch for $relativePath`: expected $($expectedHashes[$relativePath]), got $actualHash"
        }
        Write-Output "HASH OK $relativePath $actualHash"
    }

    Write-Output "RELEASE VALIDATION PASSED"
}
finally {
    Pop-Location
}
