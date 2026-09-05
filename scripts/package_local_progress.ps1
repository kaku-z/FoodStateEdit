param([string]$OutputRoot = 'artifacts/local_progress_20260905_v1')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$destination = [IO.Path]::GetFullPath((Join-Path $repo $OutputRoot))
if (Test-Path -LiteralPath $destination) { throw "Refusing to overwrite $destination" }
$checks = [Collections.Generic.List[object]]::new()
function Verify-Record($path, $expected, $group) {
    $exists = Test-Path -LiteralPath $path -PathType Leaf
    $hash = if ($exists) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    $size = if ($exists) { (Get-Item -LiteralPath $path).Length } else { $null }
    $checks.Add([ordered]@{ group=$group; path=[IO.Path]::GetRelativePath($repo,$path).Replace('\','/'); exists=$exists; size_bytes=$size; sha256=$hash; expected_sha256=$expected.sha256; passed=($exists -and $hash -eq $expected.sha256 -and $size -eq $expected.size_bytes) })
}
$execution = Get-Content (Join-Path $repo 'configs/flexible_completion_execution_v1.json') -Raw | ConvertFrom-Json
foreach ($record in @($execution.design,$execution.geometry) + @($execution.implementation.PSObject.Properties.Value)) {
    Verify-Record (Join-Path $repo $record.path) $record 'day13_frozen_implementation'
}
$datasetRoot = Join-Path $repo $execution.dataset.local_root
Verify-Record (Join-Path $datasetRoot 'dataset_manifest.json') $execution.dataset.manifest 'day13_dataset_manifest'
$dataset = Get-Content (Join-Path $datasetRoot 'dataset_manifest.json') -Raw | ConvertFrom-Json
foreach ($section in @('frozen_source_files','derived_files','metadata')) {
    foreach ($record in $dataset.$section.PSObject.Properties.Value) { Verify-Record (Join-Path $datasetRoot $record.path) $record "day13_$section" }
}
$runs = [Collections.Generic.List[object]]::new()
foreach ($relative in @('artifacts/day11_phase_action_checkpoint_sweep_gp39_retry_v2_20260901T1119Z','artifacts/day12_phase_action_isolated_checkpoint_sweep_gp38_v1')) {
    foreach ($manifestFile in Get-ChildItem (Join-Path $repo $relative) -Filter run_manifest.json -Recurse) {
        $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
        foreach ($record in $manifest.outputs) {
            Verify-Record (Join-Path $manifestFile.DirectoryName ([IO.Path]::GetFileName($record.path))) $record $relative
        }
        $conditions = @($manifest.conditions.PSObject.Properties | ForEach-Object {
            [ordered]@{name=$_.Name; frames=$_.Value.decoded_frames; outside_support_max_pixel_difference=$_.Value.outside_support_max_pixel_difference; target_support_mae=$_.Value.target_rgb_mae_inside_support_mean}
        })
        $runs.Add([ordered]@{manifest=[IO.Path]::GetRelativePath($repo,$manifestFile.FullName).Replace('\','/'); pipeline_load_count=$manifest.pipeline_load_count; status=$manifest.status; conditions=$conditions})
    }
}
$mismatchCount = @($checks | Where-Object { -not $_.passed }).Count
if ($mismatchCount -gt 0) { $checks | Where-Object { -not $_.passed } | ConvertTo-Json -Depth 6; throw "$mismatchCount local integrity failures; package not created" }
if ($runs.Count -ne 4) { throw "Expected four local sweep manifests" }
foreach ($run in $runs) {
    if ($run.pipeline_load_count -ne 1 -or $run.conditions.Count -ne 5) { throw 'Unexpected sweep structure' }
    foreach ($condition in $run.conditions) {
        if ($condition.frames -ne 21 -or $condition.outside_support_max_pixel_difference -ne 0) { throw 'Unexpected frame/protection record' }
    }
}
$figures = [ordered]@{
    '01_day5_method_grid.png'='artifacts/day5_paper_figure_v1/day5_method_grid.png'
    '02_day8_relative3d_control.png'='artifacts/day8_fork_3d_projection_v0/control_review.png'
    '03_day8_planar_raw.png'='artifacts/foodstateedit_day8_fork_3d_vace_compare_v0/day6_planar_raw_selected_frame.png'
    '04_day8_relative3d_raw.png'='artifacts/foodstateedit_day8_fork_3d_vace_compare_v0/raw_selected_frame.png'
    '05_day12_udon_lora_off.png'='artifacts/day12_phase_action_isolated_checkpoint_sweep_gp38_v1/udon_chopsticks_imagegen_pseudo_v1/lora_off_projected_review.png'
    '06_day12_udon_step32.png'='artifacts/day12_phase_action_isolated_checkpoint_sweep_gp38_v1/udon_chopsticks_imagegen_pseudo_v1/step_32_projected_review.png'
    '07_day12_spoon_lora_off.png'='artifacts/day12_phase_action_isolated_checkpoint_sweep_gp38_v1/clear_broth_spoon_imagegen_pseudo_v1/lora_off_projected_review.png'
    '08_day12_spoon_step32.png'='artifacts/day12_phase_action_isolated_checkpoint_sweep_gp38_v1/clear_broth_spoon_imagegen_pseudo_v1/step_32_projected_review.png'
    '09_day13_control.png'='artifacts/day13_3d_guided_flexible_completion_udon_v1/review/relative3d_control_contact_sheet.png'
    '10_day13_masks.png'='artifacts/day13_3d_guided_flexible_completion_udon_v1/review/topology_mask_contact_sheet.png'
}
New-Item -ItemType Directory -Path (Join-Path $destination 'figures') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $destination 'reports') | Out-Null
$copies = [Collections.Generic.List[object]]::new()
function Copy-Evidence($sourceRelative,$targetRelative) {
    $source = Join-Path $repo $sourceRelative
    $target = Join-Path $destination $targetRelative
    Copy-Item -LiteralPath $source -Destination $target
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash) { throw "Copy mismatch $target" }
    $copies.Add([ordered]@{source=$sourceRelative; packaged_path=$targetRelative; sha256=$sourceHash; size_bytes=(Get-Item -LiteralPath $target).Length})
}
foreach ($item in $figures.GetEnumerator()) { Copy-Evidence $item.Value ('figures/'+$item.Key) }
foreach ($name in @('DAY5_METHOD_GRID_CAPTION.md','DAY8_3D_PROJECTION_PILOT.md','DAY8_VACE_FORK_3D_REVIEW.md','DAY11_PHASE_ACTION_CHECKPOINT_SWEEP_RESULT.md','DAY12_PHASE_ACTION_ISOLATION_RESULT.md','DAY13_3D_GUIDED_FLEXIBLE_COMPLETION_DESIGN.md','day11_phase_action_checkpoint_sweep_result_v1.json','day12_phase_action_isolation_result_v1.json')) { Copy-Evidence ('results/'+$name) ('reports/'+$name) }
$report = Get-Content (Join-Path $repo 'results/LOCAL_PROGRESS_20260905.md') -Raw
$report.Replace('../artifacts/local_progress_20260905_v1/','./') | Set-Content (Join-Path $destination 'README.md') -Encoding utf8
$audit = [ordered]@{
    schema_version='foodstateedit.local_progress_inventory.v1'
    created_at_utc=[DateTime]::UtcNow.ToString('o')
    scope='Local byte verification against existing manifests; no new GPU execution or independent visual review'
    record_count=$checks.Count
    mismatch_count=$mismatchCount
    day13_remote_status='Power outage reported by user; latest checkpoint existence and weighted v3 completion unverified'
    checks=$checks
    sweep_manifests=$runs
    copied_assets=$copies
}
$audit | ConvertTo-Json -Depth 15 | Set-Content (Join-Path $destination 'local_integrity_audit.json') -Encoding utf8
[ordered]@{output=$destination;verified_records=$checks.Count;mismatches=$mismatchCount;copied_assets=$copies.Count;sweep_conditions=20} | ConvertTo-Json
