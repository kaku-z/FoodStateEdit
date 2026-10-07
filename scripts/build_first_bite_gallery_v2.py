"""Current all-case, all-seed gallery with geometry-specific fixed crops."""
import argparse,json,hashlib
from pathlib import Path
from zipfile import ZipFile
from fnmatch import fnmatchcase
from io import BytesIO
from PIL import Image,ImageDraw
from build_first_bite_gallery import HTML
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--default-method',default='g40_source_photo_cut');a=ap.parse_args();r=a.root;out=r/'gallery';out.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True);cases0=json.loads((r/'geometry_v3_controls/inputs/manifest.json').read_text(encoding='utf-8'))['cases'];archives={};records=[]
 def archive(name):
  if name not in archives:archives[name]=ZipFile(r/(name+'.zip'))
  return archives[name]
 def im(name,member):return Image.open(BytesIO(archive(name).read(member))).convert('RGB')
 def save(pic,name,origin):
  p=assets/name;pic.convert('RGB').save(p);records.append({'path':'assets/'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'origin':origin});return 'assets/'+name
 def fixed_box(mask,c):
  bounds=mask.convert('L').getbbox();cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2;l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];x=int(max(l,min(l+w-192,round(cx-96))));y=int(max(t,min(t+h-192,round(cy-96))));return(x,y,x+192,y+192)
 specs=[
 ('baseline','原有两阶段方法（G20 / G21）',None,None,'legacy'),
 ('g32_neural','共享直切几何 · 两阶段生成（G32）','gate_v32_complete','gate_v32/collection_complete/{cid}__box_cap_food_intrinsic__{seed}','g32'),
 ('g32_source','共享直切几何 · 原图材质（G32）','box_cap_material_derivatives_v1','gate_v32/observed_material_box_cap_uv_v1_source_grid_v2/{cid}__box_cap_food_intrinsic__{seed}','g32'),
 ('g35_neural','姿态优化 · 两阶段生成（G35）','gate_v35_complete','gate_v35/collection_complete/{cid}__observed_surface_food_intrinsic__{seed}','g35'),
 ('g35_source','姿态优化 · 切走区域原图材质（G35）','all_observed_surface_transport_consumptive_v2','gate_v35/all_observed_surface_consumptive_v2/source_rgb_unrelit/{cid}__observed_surface_food_intrinsic__{seed}','g35'),
 ('g35_light','姿态优化 · 原图材质与光照先验（G35）','all_observed_surface_transport_consumptive_v2','gate_v35/all_observed_surface_consumptive_v2/source_rgb_scalar_neural_light/{cid}__observed_surface_food_intrinsic__{seed}','g35'),
 ('g36_parent','相机配准 · 全图模型合成（G36）','gate_v36_complete','gate_v36/parent_worker_*/{cid}__source_fit_spoon_photo_canny__{seed}','g36'),
 ('g36_neural','相机配准 · 两阶段生成（G36）','gate_v36_complete','gate_v36/collection_complete/{cid}__source_fit_food_intrinsic__{seed}','g36'),
 ('g36_source','相机配准 · 切走区域原图材质（G36）','source_fit_surface_transport_consumptive_v3','gate_v36/source_fit_surface_consumptive_v3/source_rgb_unrelit/{cid}__source_fit_food_intrinsic__{seed}','g36'),
 ('g36_light','相机配准 · 原图材质与光照先验（G36）','source_fit_surface_transport_consumptive_v3','gate_v36/source_fit_surface_consumptive_v3/source_rgb_scalar_neural_light/{cid}__source_fit_food_intrinsic__{seed}','g36'),
 ('g36_body','相机配准 · 省略食物局部生成（G36）','gate_v36_source_geometry_food_ablation_v1','gate_v36/source_geometry_food_ablation_v1/{cid}__source_geometry_food__{seed}','g36'),
 ('g37_parent','勺挖几何 · 全图模型合成（G37）','gate_v37_complete','gate_v37/parent_worker_*/{cid}__source_fit_ellipsoid_spoon_photo_canny__{seed}','g37'),
 ('g37_body','勺挖几何 · 原图食物表面（G37）','gate_v37_complete','gate_v37/source_geometry_food_ablation_v1/{cid}__source_geometry_food__{seed}','g37'),
 ('g37_locked','勺挖几何 · 原图表面与模型缺口材质（G37）','gate_v37_complete','gate_v37/geometry_locked_action_v1/photographic_cut/{cid}__{seed}','g37'),
 ('g37_colorcut','勺挖几何 · 缺口颜色先验对照（G37）','gate_v37_complete','gate_v37/geometry_locked_action_v1/source_color_cut/{cid}__{seed}','g37'),
 ('g38_box','直切形状 · 光照梯度校正对照（G38）','gate_v38_complete','gate_v38/gate_v36/{cid}__{seed}','g36'),
 ('g38_scoop','勺挖形状 · 光照梯度校正对照（G38）','gate_v38_complete','gate_v38/gate_v37/{cid}__{seed}','g37')]
 for gate,arm,label in [(39,'photo_cut','表面分解与模型缺口'),(39,'locked_cut','表面分解与几何缺口'),(40,'source_photo_cut','材质边界修复 · 原图表面与模型缺口'),(40,'source_locked_cut','材质边界修复 · 原图表面与几何缺口'),(40,'band_photo_cut','材质边界修复 · 表面分解与模型缺口'),(40,'band_locked_cut','材质边界修复 · 表面分解与几何缺口')]:
  specs.append((f'g{gate}_{arm}',label+f'（G{gate}）',f'gate_v{gate}_complete',f'gate_v{gate}/{arm}/'+'{cid}__{seed}','g37'))
 for sigma in [35,60]:specs.append((f'g41_sigma{sigma}',f'材质细化 · 请求噪声 {sigma}/100；实际 '+('0.4746' if sigma==35 else '0.7188')+'（G41）','gate_v41_complete','gate_v41/collection_complete/{cid}__label_consistent_material_sigma'+str(sigma)+'__{seed}','g37'))
 specs.append(('g42_sigma100','固定区域 · 纯噪声材质生成（G42）','gate_v42_complete','gate_v42/collection_complete/{cid}__label_consistent_material_sigma100__{seed}','g37'))
 for gate,boundary in [(44,'局部材质'),(45,'完整材质边界'),(46,'连续材质置信度')]:
  for arm,label in [('illumination_only','光照投影'),('source_detail','光照投影与原图细节'),('neural_detail','光照投影与生成微纹理')]:
   specs.append((f'g{gate}_'+arm,boundary+' · '+label+f'（G{gate}）',f'gate_v{gate}_complete',f'gate_v{gate}/'+arm+'/{cid}__{seed}','g37'))
 for arm,label in [('source_diffuse','原图方向光'),('neural_diffuse','原图与模型联合方向光'),('neural_source_grain','联合方向光与内部微纹理先验')]:
  specs.append(('g47_'+arm,'食物与缺口共享光照 · '+label+'（G47）','gate_v47_complete','gate_v47/'+arm+'/{cid}__{seed}','g37'))
 for gate in [48,49]:
  for anchor in ['free','coarse_anchor']:
   for variant in ['direct_masked_raw','source_chroma_projected']:
    specs.append((f'g{gate}_'+anchor+'_'+variant,'食物与缺口联合细化 · '+('粗结构约束' if anchor=='coarse_anchor' else '自由细化')+' · '+('原始颜色' if variant=='direct_masked_raw' else '源图颜色投影')+f'（G{gate}）',f'gate_v{gate}_complete',f'gate_v{gate}/'+variant+'/{cid}__joint_action_'+anchor+'__{seed}','g37'))
 for variant in ['direct_masked_raw','source_chroma_projected']:
  specs.append(('g50_'+variant,'勺上食物与缺口放大联合细化 · '+('原始颜色' if variant=='direct_masked_raw' else '源图颜色投影')+'（G50）','gate_v50_complete','gate_v50/'+variant+'/{cid}__paired_patch_action__{seed}','g37'))
 for gate,arms in [(51,[('visibility25','缺口几何遮蔽 0.25'),('visibility50','缺口几何遮蔽 0.50'),('visibility75','缺口几何遮蔽 0.75')]),(52,[('geometry_light_only','源图正值光照与缺口遮蔽'),('paired_neural_grain','源图光照与受限生成细节')]),(53,[('geometry_soft','连续边界与平滑光照'),('source_gradient','连续边界与原图梯度'),('neural_gradient','连续边界与受限生成梯度')])]:
  for arm,label in arms:specs.append((f'g{gate}_'+arm,label+f'（G{gate}）',f'gate_v{gate}_complete',f'gate_v{gate}/'+arm+'/{cid}__{seed}','g37'))
 for variant in ['direct_masked_raw','source_chroma_projected']:
  specs.append(('g54_'+variant,'动作上下文与真实材质条件分离 · '+('原始颜色' if variant=='direct_masked_raw' else '源图颜色投影')+'（G54）','gate_v54_complete','gate_v54/'+variant+'/{cid}__real_source_conditioned_action__{seed}','g37'))
  specs.append(('g55_'+variant,'真实外观条件与粗动作约束 · '+('原始颜色' if variant=='direct_masked_raw' else '源图颜色投影')+'（G55）','gate_v55_complete','gate_v55/'+variant+'/{cid}__real_source_coarse_action__{seed}','g37'))
  specs.append(('g56_'+variant,'共享几何与独立食物、缺口生成 · '+('原始颜色' if variant=='direct_masked_raw' else '源图颜色投影')+'（G56）','gate_v56_complete','gate_v56/'+variant+'/{cid}__{seed}','g37'))
 specs=[x for x in specs if x[2] is None or (r/(x[2]+'.zip')).exists()];methods=[{'id':x[0],'label':x[1]} for x in specs];cases=[]
 for i,c in enumerate(cases0):
  cid=c['case_id'];geo=r/'box_cap_controls_v1/geometry_spoon_box_cap_v1'/cid;src=Image.open(geo/'source.png').convert('RGB');l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);row={'id':cid,'source':save(src.crop(rect),f'c{i}_source.png',str(geo/'source.png')),'outputs':{}}
  for seed in [41,163,907]:
   boxes={}
   for mi,(method,label,arc,pattern,family) in enumerate(specs):
    if arc is None:arc='gate_v20_complete' if seed==41 else 'gate_v21_complete';pattern=('gate_v20' if seed==41 else 'gate_v21')+'/collection_complete/{cid}__silken_food_only_canny__{seed}'
    folder=pattern.format(cid=cid,seed=seed)
    if '*' in folder:
     matches=[n.removesuffix('/composited.png') for n in archive(arc).namelist() if fnmatchcase(n,folder+'/composited.png')];assert len(matches)==1,(method,cid,seed);folder=matches[0]
    full=im(arc,folder+'/composited.png')
    if family not in boxes:
     if family in ['legacy','g32']:
      sb=tuple(json.loads(archive('gate_v32_complete').read(f'gate_v32/contexts/{cid}/{seed}/transform.json'))['box']);cm=im('shared_cavity_projection_v1',f'gate_v32/cavity_geometry_projection_v1/{cid}_field/cavity_mask.png')
     elif family=='g35':
      sb=tuple(json.loads(archive('gate_v35_complete').read(f'gate_v35/contexts/{cid}/{seed}/transform.json'))['box']);cm=im('observed_surface_controls_v1',f'geometry_spoon_observed_surface_v1/{cid}/source_bite_mask.png')
     elif family=='g36':
      sb=tuple(json.loads(archive('gate_v36_complete').read(f'gate_v36/contexts/{cid}/{seed}/transform.json'))['box']);cm=im('source_fit_geometry_preview_v2',f'{cid}/source_bite_mask.png')
     else:
      sb=fixed_box(im('gate_v37_complete',f'source_fit_ellipsoid_food_fields_consumptive_v4/{cid}/observed_surface_mask.png'),c);cm=im('gate_v37_complete',f'geometry_spoon_source_fit_ellipsoid_v3/{cid}/source_bite_mask.png')
     boxes[family]=(sb,fixed_box(cm,c))
    sb,cb=boxes[family];name=f'c{i}_m{mi}_s{seed}';entry={'path':save(full.crop(rect),name+'.png',arc+'.zip:'+folder+'/composited.png'),'spoon':save(full.crop(sb),name+'_food.png','Fixed source-geometry crop: '+str(sb)),'cavity':save(full.crop(cb),name+'_cut.png','Fixed source-geometry crop: '+str(cb)),'geometry_family':family,'food_crop_box':sb,'cavity_crop_box':cb}
    if method.startswith(('g41_','g42_','g44_','g45_','g46_','g47_')):
     raw_arc='gate_v41_complete' if method.startswith('g41_') else 'gate_v42_complete';gate='gate_v41' if method.startswith('g41_') else 'gate_v42';sigma=method.split('sigma')[-1] if method.startswith('g41_') else '100';pat=gate+'/worker_*/'+cid+'__label_consistent_material_sigma'+sigma+'__'+str(seed)+'/raw.png';matches=[n for n in archive(raw_arc).namelist() if fnmatchcase(n,pat)];assert len(matches)==1,(method,cid,seed);entry['raw_material']=save(im(raw_arc,matches[0]),name+'_raw.png',raw_arc+'.zip:'+matches[0])
    if method.startswith(('g48_','g49_')):
     gate=method.split('_')[0][1:];anchor='coarse_anchor' if '_coarse_anchor_' in method else 'free';pat=f'gate_v{gate}/worker_*/'+cid+'__joint_action_'+anchor+'__'+str(seed)+'/raw.png';matches=[n for n in archive(f'gate_v{gate}_complete').namelist() if fnmatchcase(n,pat)];assert len(matches)==1;entry['raw_material']=save(im(f'gate_v{gate}_complete',matches[0]),name+'_raw.png',f'gate_v{gate}_complete.zip:'+matches[0])
    if method.startswith('g50_') or method in ['g52_paired_neural_grain','g53_neural_gradient']:
     pat='gate_v50/worker_*/'+cid+'__paired_patch_action__'+str(seed)+'/raw.png';matches=[n for n in archive('gate_v50_complete').namelist() if fnmatchcase(n,pat)];assert len(matches)==1;entry['raw_material']=save(im('gate_v50_complete',matches[0]),name+'_raw.png','gate_v50_complete.zip:'+matches[0])
     entry['raw_caption']='模型原始输出：左侧放大的勺上食物，右侧放大的盘中缺口；未经坐标恢复和最终合成。'
    if method.startswith('g54_'):
     pat='gate_v54/worker_*/'+cid+'__real_source_conditioned_action__'+str(seed)+'/raw.png';matches=[n for n in archive('gate_v54_complete').namelist() if fnmatchcase(n,pat)];assert len(matches)==1;entry['raw_material']=save(im('gate_v54_complete',matches[0]),name+'_raw.png','gate_v54_complete.zip:'+matches[0]);entry['raw_caption']='模型原始输出：左侧放大的勺上食物，右侧放大的盘中缺口；真实原图提供外观条件，动作布局用于固定上下文，未经最终合成。'
    if method.startswith('g55_'):
     pat='gate_v55/worker_*/'+cid+'__real_source_coarse_action__'+str(seed)+'/raw.png';matches=[n for n in archive('gate_v55_complete').namelist() if fnmatchcase(n,pat)];assert len(matches)==1;entry['raw_material']=save(im('gate_v55_complete',matches[0]),name+'_raw.png','gate_v55_complete.zip:'+matches[0]);entry['raw_caption']='模型原始输出：左侧放大的勺上食物，右侧放大的盘中缺口；真实外观条件与粗动作约束共同参与去噪，未经最终合成。'
    if method.startswith('g56_'):
     for node,key in [('food','raw_material'),('cut','raw_cut')]:
      pat='gate_v56/worker_*/'+cid+'__node_'+node+'__'+str(seed)+'/raw.png';matches=[n for n in archive('gate_v56_complete').namelist() if fnmatchcase(n,pat)];assert len(matches)==1;entry[key]=save(im('gate_v56_complete',matches[0]),name+'_raw_'+node+'.png','gate_v56_complete.zip:'+matches[0])
     entry['raw_caption']='食物节点的独立模型原始输出，未经坐标恢复与最终合成。'
    row['outputs'][method+'_'+str(seed)]=entry
  cases.append(row)
 default=a.default_method;assert default in [m['id'] for m in methods];data={'default_method':default,'cases':cases,'methods':methods,'scope':'Development cases only; no true before/after pair, measured 3D, heldout generalization or blind human evaluation. Final compositions are distinct from raw foundation images.'}
 (out/'manifest.json').write_text(json.dumps(dict(data,assets=records),ensure_ascii=False,indent=2),encoding='utf-8');html=HTML.replace("el('method').value='source_material';","el('method').value=data.default_method;")
 html=html.replace('显示固定几何区域</label></nav>','显示固定几何区域</label><label><input type="checkbox" id="raw">显示模型原始输出</label></nav>')
 html=html.replace('card.append(d)}el(\'cases\').append(card)',"card.append(d)}if(el('raw').checked&&a?.raw_material){const f=document.createElement('figure');f.className='frame';const im=document.createElement('img');im.src=a.raw_material;im.alt='模型原始输出，未经后续合成';im.style.maxHeight='420px';const cap=document.createElement('figcaption');cap.textContent='模型原始输出，未经后续合成；它与上方最终合成图是不同阶段。';f.append(im,cap);card.append(f)}el('cases').append(card)")
 html=html.replace("['case','seed','method','detail']","['case','seed','method','detail','raw']")
 html=html.replace("cap.textContent='模型原始输出，未经后续合成；它与上方最终合成图是不同阶段。';","cap.textContent=a.raw_caption||'模型原始输出，未经后续合成；它与上方最终合成图是不同阶段。';")
 html=html.replace("card.append(f)}el('cases').append(card)","card.append(f)}if(el('raw').checked&&a?.raw_cut){const f=document.createElement('figure');f.className='frame';const im=document.createElement('img');im.src=a.raw_cut;im.alt='缺口节点的独立模型原始输出';im.style.maxHeight='420px';const cap=document.createElement('figcaption');cap.textContent='缺口节点的独立模型原始输出；它与食物节点共用几何和材质条件，未经最终合成。';f.append(im,cap);card.append(f)}el('cases').append(card)")
 (out/'index.html').write_text(html.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
 board=Image.new('RGB',(1200,2560),'white');draw=ImageDraw.Draw(board)
 for i,c in enumerate(cases):
  for j,p in enumerate([c['source'],c['outputs']['baseline_41']['path'],c['outputs'][default+'_41']['path']]):
   pic=Image.open(out/p);pic.thumbnail((400,278));board.paste(pic,(j*400+(400-pic.width)//2,i*320+32));draw.text((j*400+6,i*320+7),['source','previous seed41',default+' seed41'][j]+' '+c['id'],fill='black')
 board.save(out/'all_cases_seed41.jpg',quality=95)
 assert len(cases)==8 and all(len(c['outputs'])==3*len(specs) for c in cases)
 for record in records:
  with Image.open(out/record['path']) as pic:pic.verify()
 print('CURRENT_GALLERY',len(cases),len(specs),sum(len(c['outputs']) for c in cases),len(records),flush=True)
 for z in archives.values():z.close()
if __name__=='__main__':main()
