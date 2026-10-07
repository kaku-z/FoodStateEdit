"""Plot all frozen conditions after review, with explicit denominators and no inference claims."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    s = json.loads(a.summary.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=True)
    dims = ['lift', 'utensil_support', 'matching_source_notch', 'scene_identity', 'no_person', 'photographic_realism']
    labels = ['Lift', 'Utensil\nsupport', 'Source\nnotch', 'Scene\nidentity', 'No\nperson', 'Photo\nrealism', 'All six']
    rows = s['summary']
    values = np.array([[r['dimensions'][d]['pass'] for d in dims] + [r['complete_successes']] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 6.2), layout='constrained')
    im = ax.imshow(values, vmin=0, vmax=24, cmap='Blues', aspect='auto')
    ax.set_xticks(range(7), labels)
    ax.set_yticks(range(8), [r['backend'].upper() + ' / ' + r['method'].split('_')[0] for r in rows])
    for (y, x), n in np.ndenumerate(values):
        ax.text(x, y, f'{n}/24', ha='center', va='center', color='white' if n >= 16 else '#172b4d', fontsize=12)
    ax.axhline(3.5, color='white', linewidth=4)
    ax.axvline(5.5, color='white', linewidth=4)
    ax.set_title('First-bite diagnostic review: all intended conditions', loc='left', pad=15)
    fig.colorbar(im, ax=ax, label='Pass count (24 intended outputs per row)', shrink=.75)
    fig.supxlabel('Assistant review, unblinded; uncertain and missing outputs are nonpasses.\n8 curated photographs, 3 paired seeds; not 24 independent photographs.', fontsize=10)
    fig.savefig(a.output/'diagnostic_pass_counts.png', dpi=180)
    fig.savefig(a.output/'diagnostic_pass_counts.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained', sharey=True)
    for ax, backend in zip(axes, ['qwen', 'vace']):
        ps = [r for r in s['photo_averages'] if r['backend'] == backend]
        cases = sorted({r['case_id'] for r in ps})
        methods = ['A_direct', 'B_planar', 'C_rgb3d', 'D_staged3d']
        matrix = np.array([[next(r['success_rate'] for r in ps if r['case_id'] == c and r['method'] == m) for m in methods] for c in cases])
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Greens', aspect='auto')
        for (y, x), v in np.ndenumerate(matrix):
            ax.text(x, y, f'{round(v*3)}/3', ha='center', va='center', color='white' if v > .6 else '#172b4d')
        ax.set_xticks(range(4), ['A direct', 'B planar', 'C RGB 3D', 'D staged 3D'], rotation=20, ha='right')
        ax.set_yticks(range(8), [c.replace('test_', '') for c in cases])
        ax.set_title(backend.upper())
    fig.suptitle('Complete success by source photograph (all six criteria)', x=.02, ha='left')
    fig.supxlabel('Assistant diagnostic review; 06 has preprocessing failures for B/C/D.\nPhotographs 03 and 06 share a conservative scene cluster.', fontsize=10)
    fig.savefig(a.output/'per_photo_complete_success.png', dpi=180)
    fig.savefig(a.output/'per_photo_complete_success.pdf')
    plt.close(fig)

    # Fixed first source and first seed, with every method retained.
    cells = s['cells']
    case = 'test_01_7445'
    seed = 41
    selected = [r for r in cells if r['case_id'] == case and r['seed'] == seed]
    assert len(selected) == 8
    canvas = Image.new('RGB', (2560, 1092), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    draw.text((12, 8), 'Fixed example: photograph 01 (7445), seed 41. All raw outputs; no best-seed selection.', fill='black', font=font)
    for y, backend in enumerate(['qwen', 'vace']):
        for x, method in enumerate(['A_direct', 'B_planar', 'C_rgb3d', 'D_staged3d']):
            r = next(r for r in selected if r['backend'] == backend and r['method'] == method)
            label = backend.upper() + ' / ' + method + ' | ' + str(r['success']) + ' complete success'
            draw.text((x*640+8, 52+y*520), label, fill='black', font=font)
            rel = r['raw_path'].split('/first_bite_complete_20260929/', 1)[1]
            img = Image.open(a.bundle/rel).convert('RGB')
            assert img.size == (640, 480)
            canvas.paste(img, (x*640, 88+y*520))
    canvas.save(a.output/'fixed_case01_seed41.jpg', quality=95)
    (a.output/'figure_manifest.json').write_text(json.dumps({
        'data': str(a.summary), 'review': s['reviewer'], 'primary_denominator': 192,
        'example_selection': 'Fixed first photograph 01, first seed 41, all methods and both backends; descriptive illustration, not rate evidence',
        'selected_raw_sha256': {r['backend']+'/'+r['id']: r['raw_sha256'] for r in selected},
        'plot_limits': 'Descriptive assistant diagnostic counts; no statistical significance or population generalization claim.'
    }, indent=2)+'\n', encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
