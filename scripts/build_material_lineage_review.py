"""Package all fixed material-lineage pilot cells for direct review."""
import argparse,hashlib,html,json,shutil,zipfile
from pathlib import Path
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build(root, no_archive=False):
    config=json.loads((root/'config.json').read_text());cases=config['cases'];seeds=config['seeds']
    rows=[];boards=[];source_rows=[]
    for cid in cases:
        d=root/cid;manifest=json.loads((d/'manifest.json').read_text());assert len(manifest['results'])==18
        inputs=json.loads((d/'source_inputs.json').read_text())
        snapshot=d/'input_snapshot';snapshot.mkdir(exist_ok=True)
        copied=[]
        for i,p in enumerate(inputs['inputs']):
            src=Path(p['path'])
            if src.exists():
                dst=snapshot/(str(i)+'_'+src.name)
                if not dst.exists():shutil.copy2(src,dst)
                assert sha(dst)==p['sha256'];copied.append({'original_path':str(src),'snapshot':str(dst.relative_to(root)),'sha256':sha(dst)})
        (snapshot/'index.json').write_text(json.dumps(copied,indent=2),encoding='utf-8')
        rect=tuple(manifest['results'][0]['native_rect']);src=Image.open(d/'source.png').convert('RGB').crop(rect)
        tilew=320;tileh=round(tilew*src.height/src.width)
        for stage in ['v1','v2']:
            if stage=='v2' and not (root/'metal_v2'/cid/'pose0/shared_s41/composited.png').exists():continue
            board=Image.new('RGB',(tilew*6,3*(tileh+23)+70),'#e9edf0');draw=ImageDraw.Draw(board)
            draw.text((10,7),f'{cid} | {stage} | columns shared s41,s163,s907 / independent s41,s163,s907; rows poses 0,1,2',fill='black')
            thumb=src.copy();thumb.thumbnail((90,50));board.paste(thumb,(10,25))
            rawboard=Image.new('RGB',(tilew*6,3*(240+23)+28),'#e9edf0');rd=ImageDraw.Draw(rawboard)
            rd.text((8,6),f'{cid} | raw 3D render v1, same fixed matrix',fill='black')
            for pose in range(3):
                for j,variant in enumerate(['shared','independent']):
                    for k,seed in enumerate(seeds):
                        base=d/f'pose{pose}'/f'{variant}_s{seed}'
                        final=base/'composited.png' if stage=='v1' else root/'metal_v2'/cid/f'pose{pose}'/f'{variant}_s{seed}'/'composited.png'
                        assert final.exists();im=Image.open(final).convert('RGB').crop(rect).resize((tilew,tileh),Image.Resampling.LANCZOS)
                        x=(j*3+k)*tilew;y=70+pose*(tileh+23)
                        draw.text((x+5,y+3),f'{variant} pose{pose} seed{seed}',fill='black');board.paste(im,(x,y+23))
                        raw=Image.open(base/'render.png').convert('RGB').resize((tilew,240),Image.Resampling.LANCZOS)
                        yr=28+pose*263;rd.text((x+5,yr+3),f'{variant} pose{pose} seed{seed}',fill='black');rawboard.paste(raw,(x,yr+23))
                        rows.append({'case_id':cid,'stage':stage,'pose':pose,'seed':seed,'variant':variant,'path':str(final.relative_to(root)),'sha256':sha(final)})
            board.save(d/f'review_{stage}.jpg',quality=95)
            if stage=='v1':rawboard.save(d/'review_raw_v1.jpg',quality=95)
            boards.append({'case_id':cid,'stage':stage,'path':str((d/f'review_{stage}.jpg').relative_to(root))})
        source_rows.append((cid,src,tileh,rect))
    stage='v2' if len([r for r in rows if r['stage']=='v2'])==144 else 'v1'
    width=1280;total=sum(th+26 for _,_,th,_ in source_rows)+40
    board=Image.new('RGB',(width,total),'white');draw=ImageDraw.Draw(board);draw.text((8,8),'Real source | same material, raised poses 0 / 1 / 2 (fixed seed 41)',fill='black');y=35
    for cid,src,th,rect in source_rows:
        for col in range(4):
            if col==0:im=src
            else:
                base=(root/cid if stage=='v1' else root/'metal_v2'/cid)/f'pose{col-1}'/'shared_s41/composited.png'
                im=Image.open(base).convert('RGB').crop(rect)
            im=im.resize((320,th),Image.Resampling.LANCZOS);board.paste(im,(col*320,y+22))
            draw.text((col*320+5,y+4),cid if col==0 else f'pose{col-1}, seed41',fill='black')
        y+=th+26
    board.save(root/'comparison_all_sources.jpg',quality=96)
    data={'config':config,'rows':rows,'boards':boards,'default_stage':stage}
    page='''<!doctype html><meta charset="utf-8"><title>Material lineage pilot</title><style>
    body{margin:0;background:#111721;color:#eaf0f5;font:16px system-ui}main{max-width:1500px;margin:auto;padding:22px}select{padding:8px;margin:5px;background:#253043;color:white}img{max-width:100%;background:white}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cell small{display:block}a{color:#93caff}.source{max-height:350px}p{line-height:1.6}.note{color:#ffcc8e}</style><main>
    <h1>物质谱系：真实原图小型验证</h1><p>8 张真实原图；每张实际训练 6,000 步，57,987 参数；三个固定随机种子、三个抬升位置、共享与独立材质对照。全部固定输出均可查看。</p>
    <p class="note">仅为物质身份、切面与跨姿态一致性原型。形状来自单图拟合，训练是各原图自己的材质小模型，缺口、材质和餐具仍有 CG 感；未证明真实物理、照片真实性、完整新模型或跨食物泛化。</p>
    <label>原图 <select id="case"></select></label><label>方法 <select id="variant"><option>shared</option><option>independent</option></select></label><label>渲染 <select id="stage"></select></label>
    <p id="label"></p><img class="source" id="source"><p>行：抬升位置 0 / 1 / 2；列：seed 41 / 163 / 907。点击打开实际分辨率输出。</p><div class="grid" id="grid"></div>
    <p><a href="audit.json">独立代码审计</a> · <a href="RESULTS.html">结果与边界</a> · <a href="config.json">冻结计划</a> · <a href="comparison_all_sources.jpg">八张原图总览</a></p></main><script>const DATA=__DATA__;
    const c=document.getElementById('case'),v=document.getElementById('variant'),s=document.getElementById('stage');DATA.config.cases.forEach(x=>c.add(new Option(x,x)));['v1','v2'].filter(x=>DATA.rows.some(r=>r.stage===x)).forEach(x=>s.add(new Option(x,x)));s.value=DATA.default_stage;
    function render(){document.getElementById('source').src=c.value+'/source.png';document.getElementById('label').textContent=c.value+' — '+v.value+' — '+s.value;const g=document.getElementById('grid');g.innerHTML='';DATA.rows.filter(r=>r.case_id===c.value&&r.variant===v.value&&r.stage===s.value).sort((a,b)=>a.pose-b.pose||DATA.config.seeds.indexOf(a.seed)-DATA.config.seeds.indexOf(b.seed)).forEach(r=>{const d=document.createElement('div');d.className='cell';const a=document.createElement('a');a.href=r.path;a.target='_blank';const im=document.createElement('img');im.src=r.path;a.append(im);const cap=document.createElement('small');cap.textContent='pose '+r.pose+' · seed '+r.seed;d.append(a,cap);g.append(d)});}c.onchange=v.onchange=s.onchange=render;render();</script>'''.replace('__DATA__',json.dumps(data,ensure_ascii=False))
    (root/'gallery.html').write_text(page,encoding='utf-8')
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix not in ['.zip','.tmp'] and '__pycache__' not in p.parts]
    manifest={'status':'complete_unreviewed','v1_images':144,'v2_images':len([r for r in rows if r['stage']=='v2']),
              'rows':rows,'boards':boards,'files':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if no_archive:
        print('REVIEW_READY',len(files),'files',len(rows),'photos',flush=True)
        return
    archive=Path('/mnt/tmp/material_lineage_pilot_20261003.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'artifact_manifest.json']:z.write(p,p.relative_to(root))
    print('REVIEW_PACKAGED',len(files),'files',len(rows),'photo outputs',archive.stat().st_size,'bytes',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/host/space0/guo-z/tf-ufi/material_lineage_pilot_20261003'));ap.add_argument('--no-archive',action='store_true');a=ap.parse_args();build(a.root,a.no_archive)
