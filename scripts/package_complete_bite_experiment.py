"""Package the verified experiment, including every raw result and review evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser


def main():
    root = Path('outputs/first_bite_complete_20260929')
    audit = json.loads((root/'analysis/delivery_audit.json').read_text())
    assert audit['status'] == 'verified_compute_and_assistant_review_complete'
    report = Path('docs/COMPLETE_FIRST_BITE_20260929.md').read_text(encoding='utf-8')
    navigation = '''
[完整原图图库：192 个条件](review/all_results.html) · [逐图诊断 CSV](review/assistant_review.csv) · [全部统计](analysis/summary.json) · [交付校验](analysis/delivery_audit.json)

下图保留每个方法的全部预定分母。A=直接编辑，B=二维贴片，C=三维代理细化，D=C 加分阶段约束。

![六项诊断通过数](figures/diagnostic_pass_counts.png)

'''
    title, body = report.split('\n', 1)
    appendix = '''
## 查看与重算

- `review/all_results.html`：192 个条件全部保留，左侧原图、右侧未经后处理的输出；失败条件也在其中。按 R 编号可查询逐图 CSV。方法名默认折叠，不代表已有盲评。
- `review/human_review_blinded.csv`：未填写的人类复核表，不含方法信息；本次没有收集人类评分。
- `assistant_review_notes.json`、`review_pages*`：助手逐图诊断笔记和当时检查的全尺寸拼页。
- `analysis/summary.json`：全部六项分类、照片和场景簇配对比较、噪声一致性和统计误读检查。
- `analysis/delivery_audit.json`：冻结摘要、原图、输出、复跑、检测与资源核验。
- `server_results/`：计算档案；`formal/FROZEN.json` 为正式生成前的内部冻结记录，`provenance/` 为依赖源码与环境清单。
- `selection_evidence/`：原图筛选拼页、候选图包与几何控制检查图。
- `analysis_code/`：准备、运行、审查和分析脚本的交付副本。
- `DELIVERY_MANIFEST.json`：本交付内所有文件的 SHA-256；压缩包自身摘要另存在同名 receipt JSON。

离线查看可直接打开图库 HTML。下列命令在解压后的根目录重算现有评分的统计与图表，不调用服务器模型，也不会补造评分：

```powershell
py -3.13 analysis_code/summarize_complete_bite_experiment.py --bundle server_results --review review/assistant_review.csv --output analysis_recomputed
py -3.13 analysis_code/plot_complete_bite_experiment.py --bundle server_results --summary analysis_recomputed/summary.json --output figures_recomputed
py -3.13 analysis_code/audit_complete_bite_delivery.py --bundle server_results --review assistant_review_notes.json --output analysis_recomputed/delivery_audit.json
```

分析需要 NumPy、Pillow 和 matplotlib；审查图库生成脚本还使用 Windows Arial 字体路径。生成脚本依赖原服务器各环境及绝对模型/输入路径，材料中提供了关键依赖源码摘要和环境包清单，但没有容器镜像或模型权重，因此不是任意机器解压后一键重跑的发行版。另行复跑模型时使用新输出目录并保留原冻结运行。

## 固定示例与逐照片结果

下面固定展示编号 01、种子 41 的八个原始输出；采用第一张图、第一个预定种子，未挑选最佳结果。示例不替代全部条件的统计。

![固定照片和种子的全部方法](figures/fixed_case01_seed41.jpg)

![逐照片完整通过数量](figures/per_photo_complete_success.png)
'''
    (root/'REPORT.md').write_text(title+'\n'+navigation+body+appendix, encoding='utf-8')
    code_dir = root/'analysis_code'
    code_dir.mkdir(exist_ok=True)
    for p in Path('scripts').glob('*complete_bite*.py'):
        shutil.copy2(p, code_dir/p.name)
    selection = root/'selection_evidence'
    selection.mkdir(exist_ok=True)
    for name in ['candidate_sheet.jpg', 'candidate_sources.zip', 'geometry_source_review.jpg']:
        shutil.copy2(root/name, selection/name)

    class GalleryParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = []
            self.cards = 0
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'img':
                self.images.append(attrs['src'])
            if tag == 'article':
                self.cards += 1
    parser = GalleryParser()
    parser.feed((root/'review/all_results.html').read_text(encoding='utf-8'))
    assert parser.cards == 192 and len(parser.images) == 366
    assert all((root/'review'/rel).is_file() for rel in parser.images)
    dirs = ['server_results', 'review', 'analysis', 'figures', 'analysis_code', 'selection_evidence']
    dirs += sorted(p.name for p in root.glob('review_pages*') if p.is_dir())
    files = [root/'REPORT.md', root/'assistant_review_notes.json']
    files += [p for d in dirs for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    def sha(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(16*1024**2), b''):
                h.update(block)
        return h.hexdigest()
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'formal_conditions': 192, 'generated_outputs': 174, 'preprocessing_failures': 18,
        'review': 'One unblinded assistant diagnostic review; no independent human ratings',
        'gallery': {'cards': parser.cards, 'image_references_verified': len(parser.images)},
        'files': {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)}}
    mp = root/'DELIVERY_MANIFEST.json'
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    archive = root/'first_bite_experiment_complete_v1.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(files+[mp]):
            z.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)+1
    receipt = {'status': 'complete_verified', 'archive': archive.name, 'size_bytes': archive.stat().st_size,
        'sha256': sha(archive), 'file_count': len(files)+1, 'manifest_sha256': sha(mp),
        'excluded': 'Model weights, raw NPZ arrays and redundant progress snapshots; originals remain on server/local workspace.'}
    (root/'first_bite_experiment_complete_v1_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
