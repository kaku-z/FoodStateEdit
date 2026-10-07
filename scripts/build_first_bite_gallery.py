"""Build an offline all-case gallery from preserved experiment archives."""
import argparse,hashlib,json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageDraw

HTML=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一口食物 · 实验对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#202a35;font:15px/1.6 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:29px;margin:0}p{margin:8px 0 16px;color:#566374}header{padding:22px;background:white;border-radius:14px}.status{display:inline-block;background:#fff1d6;color:#79510a;padding:3px 12px;border-radius:20px;margin:10px 0}nav{position:sticky;top:0;z-index:2;padding:14px 0;background:#f4f5f7;display:flex;gap:16px;flex-wrap:wrap}select{padding:9px;border:1px solid #bdc6d1;border-radius:7px;font-size:15px}label{display:flex;gap:8px;align-items:center}.cases{display:grid;gap:20px}.case{background:white;border-radius:12px;padding:18px}.case h2{margin:0 0 12px;font-size:18px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.frame{margin:0;min-width:0}img{width:100%;display:block;object-fit:contain;background:#f1f1f1;max-height:590px}figcaption{padding:7px 0;color:#647181}.details{display:flex;gap:10px;margin-top:10px}.details img{width:180px;height:180px;object-fit:contain}.details figure{margin:0}.note{font-size:13px;color:#697585}.missing{padding:50px;background:#f4f4f4}.hidden{display:none}footer{margin:25px 0;color:#607080}@media(max-width:720px){main{padding:12px}.pair{grid-template-columns:1fr}h1{font-size:24px}.details img{width:140px;height:140px}}
</style></head><body><main><header><h1>第一口食物 · 实验对照</h1><div class="status">研究版本 · 尚未通过最终真实性验收</div><p>8 张真实输入照片。查看全部案例和固定随机种子，检查第一口是否离盘、缺口是否对应、食物是否保持原图材质。</p><div class="note">全部案例均为已反复使用的开发数据；无真实前后配对照片、扫描或独立人工评分。图中结果为算法合成，几何约束来自单图推断和先验。</div></header><nav><label>案例<select id="case"><option value="all">全部 8 张</option></select></label><label>种子<select id="seed"><option>41</option><option>163</option><option>907</option></select></label><label>处理方法<select id="method"></select></label><label><input type="checkbox" id="detail">显示固定几何区域</label></nav><section class="cases" id="cases"></section><footer>源图与结果使用同一原始内容区域显示。局部图使用生成前确定的几何区域；不按结果挑选裁剪。原始生成、合成图和确定性后处理分开保存。<br><a href="manifest.json">查看图片索引与文件哈希</a></footer></main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const el=id=>document.getElementById(id);
for(const c of data.cases){const o=document.createElement('option');o.value=c.id;o.textContent=c.id;el('case').append(o)}
for(const m of data.methods){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;el('method').append(o)}el('method').value='source_material';
function render(){el('cases').replaceChildren();for(const c of data.cases){if(el('case').value!=='all'&&el('case').value!==c.id)continue;const key=el('method').value+'_'+el('seed').value;const a=c.outputs[key];const card=document.createElement('article');card.className='case';const title=document.createElement('h2');title.textContent=c.id;card.append(title);const pair=document.createElement('div');pair.className='pair';for(const [label,path]of [['真实输入',c.source],[data.methods.find(m=>m.id===el('method').value).label+' · seed '+el('seed').value,a?.path]]){const f=document.createElement('figure');f.className='frame';if(path){const im=document.createElement('img');im.src=path;im.alt=label;im.loading='lazy';f.append(im)}else{const d=document.createElement('div');d.className='missing';d.textContent='此对照只有 seed 41；请切换种子。';f.append(d)}const cap=document.createElement('figcaption');cap.textContent=label;f.append(cap);pair.append(f)}card.append(pair);if(el('detail').checked&&a?.spoon){const d=document.createElement('div');d.className='details';for(const [label,path]of [['抬起的第一口',a.spoon],['盘中缺口',a.cavity]]){if(!path)continue;const f=document.createElement('figure');const im=document.createElement('img');im.src=path;im.alt=label;const cap=document.createElement('figcaption');cap.textContent=label;f.append(im,cap);d.append(f)}card.append(d)}el('cases').append(card)}}
for(const id of ['case','seed','method','detail'])el(id).addEventListener('change',render);render();
</script></body></html>'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;out=r/'gallery';out.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True)
    case_list=json.loads((r/'geometry_v3_controls/inputs/manifest.json').read_text())['cases']
    archives={}
    for name in ['gate_v20_complete','gate_v21_complete','gate_v32_complete','box_cap_material_derivatives_v1','shared_cavity_projection_v1','gate_v33_complete','gate_v34_complete','gate_v35_complete','all_observed_surface_transport_v1','all_observed_surface_transport_consumptive_v2']:
        p=r/(name+'.zip')
        if p.exists():archives[name]=ZipFile(p)
    records=[]
    def save_image(im,name,origin):
        p=assets/name;im.convert('RGB').save(p);records.append({'path':'assets/'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'origin':origin});return 'assets/'+name
    def image_from(archive,name):return Image.open(BytesIO(archives[archive].read(name))).convert('RGB')
    methods=[{'id':'baseline','label':'原有两阶段方法（G20 / G21）'},{'id':'box_cap_neural','label':'共享切割几何 + 两阶段生成（G32）'},{'id':'source_material','label':'G32 + 原图表面映射与原图尺度采样'},{'id':'cavity_prior','label':'缺口几何与遮挡投影对照'}]
    if 'gate_v33_complete' in archives:methods += [{'id':'cavity_free','label':'缺口表面细化 · 无内部投影（G33）'},{'id':'cavity_anchored','label':'缺口表面细化 · 低频投影（G33）'}]
    if 'gate_v35_complete' in archives:methods += [{'id':'pose_neural','label':'姿态优化 · 两阶段生成（G35）'}]
    for archive,base,prefix,label in [('all_observed_surface_transport_v1','all_observed_surface_v1','pose_source','姿态优化 · 原图表面'),('all_observed_surface_transport_consumptive_v2','all_observed_surface_consumptive_v2','pose_consumptive','原图切走区域约束')]:
        if archive in archives:
            methods += [{'id':prefix+'_raw','label':label+' · 原始光照'},{'id':prefix+'_light','label':label+' · 光照调整'}]
    cases=[]
    for index,c in enumerate(case_list):
        cid=c['case_id'];geo=r/'box_cap_controls_v1/geometry_spoon_box_cap_v1'/cid
        source=Image.open(geo/'source.png').convert('RGB');l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        row={'id':cid,'source':save_image(source.crop(rect),f'c{index}_source.png',str(geo/'source.png')),'outputs':{}}
        for seed in [41,163,907]:
            jid=cid+'__box_cap_food_intrinsic__'+str(seed)
            options=[('baseline','gate_v20_complete' if seed==41 else 'gate_v21_complete',('gate_v20' if seed==41 else 'gate_v21')+'/collection_complete/'+cid+'__silken_food_only_canny__'+str(seed)),('box_cap_neural','gate_v32_complete','gate_v32/collection_complete/'+jid),('source_material','box_cap_material_derivatives_v1','gate_v32/observed_material_box_cap_uv_v1_source_grid_v2/'+jid),('cavity_prior','shared_cavity_projection_v1','gate_v32/cavity_geometry_projection_v1/'+jid)]
            if seed==41 and 'gate_v33_complete' in archives:options += [('cavity_free','gate_v33_complete','gate_v33/collection_complete/'+cid+'__cavity_unanchored__41'),('cavity_anchored','gate_v33_complete','gate_v33/collection_complete/'+cid+'__cavity_low_frequency_projected__41')]
            if seed!=41 and 'gate_v34_complete' in archives:options += [('cavity_free','gate_v34_complete','gate_v34/collection_complete/'+cid+'__cavity_unanchored__'+str(seed))]
            new_jid=cid+'__observed_surface_food_intrinsic__'+str(seed)
            if 'gate_v35_complete' in archives:options += [('pose_neural','gate_v35_complete','gate_v35/collection_complete/'+new_jid)]
            for archive,base,prefix in [('all_observed_surface_transport_v1','all_observed_surface_v1','pose_source'),('all_observed_surface_transport_consumptive_v2','all_observed_surface_consumptive_v2','pose_consumptive')]:
                if archive in archives:options += [(prefix+'_raw',archive,'gate_v35/'+base+'/source_rgb_unrelit/'+new_jid),(prefix+'_light',archive,'gate_v35/'+base+'/source_rgb_scalar_neural_light/'+new_jid)]
            transform_name='gate_v32/contexts/'+cid+'/'+str(seed)+'/transform.json'
            trans=json.loads(archives['gate_v32_complete'].read(transform_name));spoon_box=tuple(trans['box'])
            mask=Image.open(BytesIO(archives['shared_cavity_projection_v1'].read('gate_v32/cavity_geometry_projection_v1/'+cid+'_field/cavity_mask.png'))).convert('L')
            bounds=mask.getbbox();cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2
            x0=int(max(l,min(l+w-192,round(cx-96))));y0=int(max(t,min(t+h-192,round(cy-96))));cavity_box=(x0,y0,x0+192,y0+192)
            for method,archive,folder in options:
                full=image_from(archive,folder+'/composited.png')
                name=f'c{index}_{method}_{seed}';entry={'path':save_image(full.crop(rect),name+'.png',archive+'.zip:'+folder+'/composited.png')}
                selected_spoon_box=spoon_box
                if method.startswith('pose_'):
                    selected_spoon_box=tuple(json.loads(archives['gate_v35_complete'].read('gate_v35/contexts/'+cid+'/'+str(seed)+'/transform.json'))['box'])
                entry['spoon']=save_image(full.crop(selected_spoon_box),name+'_spoon.png','source-geometry-selected fixed crop')
                entry['cavity']=save_image(full.crop(cavity_box),name+'_cavity.png','shared source geometry cavity bounds; fixed per case')
                row['outputs'][method+'_'+str(seed)]=entry
        cases.append(row)
    data={'cases':cases,'methods':methods,'scope':'All eight are development cases. No measured 3D, paired before/after truth, blind human ratings or heldout generalization. Composites and postprocessing are disclosed separately from raw model outputs.'}
    (out/'manifest.json').write_text(json.dumps(dict(data,assets=records),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'index.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    board=Image.new('RGB',(3*400,8*320),'white');draw=ImageDraw.Draw(board)
    for i,row in enumerate(cases):
        paths=[row['source'],row['outputs']['baseline_41']['path'],row['outputs']['source_material_41']['path']]
        for j,p in enumerate(paths):
            im=Image.open(out/p);im.thumbnail((400,278));board.paste(im,(j*400+(400-im.width)//2,i*320+32));draw.text((j*400+8,i*320+9),['source','previous seed41','G32 source material seed41'][j]+' '+row['id'],fill='black')
    board.save(out/'all_cases_seed41.jpg',quality=95)
    assert len(cases)==8 and all(len(c['outputs'])>=12 for c in cases)
    for record in records:
        with Image.open(out/record['path']) as im:im.verify()
    print('GALLERY',len(cases),sum(len(c['outputs']) for c in cases),len(records),flush=True)
    for archive in archives.values():archive.close()
if __name__=='__main__':main()
