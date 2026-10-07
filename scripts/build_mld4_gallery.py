"""Build inspectable MLD4 comparison/contact/detail sheets from evaluated files."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def h(value):
    return html.escape(str(value))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return 'unmeasured' if value is None else f'{value:.5g}'


def href(path, out):
    return os.path.relpath(path,out).replace(os.sep,'/')


def figure(path,label,out):
    if path is None or not path.exists():
        return f'<figure><div class="missing">Not available</div><figcaption>{h(label)}</figcaption></figure>'
    link=h(href(path,out))
    return f'<figure><a href="{link}"><img loading="lazy" src="{link}" alt="{h(label)}"></a><figcaption>{h(label)}</figcaption></figure>'


def fit_image(path,size):
    if path and path.exists():
        im=Image.open(path).convert('RGB'); im.thumbnail(size)
        return im
    return None


def font(size):
    try:
        return ImageFont.truetype('arial.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def bbox_from_masks(folder,names,pad=20):
    masks=[np.asarray(Image.open(folder/name).convert('L'))>0 for name in names if (folder/name).exists()]
    if not masks:
        return None
    m=np.logical_or.reduce(masks); y,x=np.where(m)
    if not len(x):
        return None
    return (max(0,int(x.min())-pad),max(0,int(y.min())-pad),
            min(m.shape[1],int(x.max())+pad+1),min(m.shape[0],int(y.max())+pad+1))


def make_detail_sheet(row,out):
    folder=Path(row['folder'])
    removal_box=bbox_from_masks(folder,['source_removed_mask.png'],20)
    target_box=bbox_from_masks(folder,['moved_food_mask.png'],24)
    panels=[('Source removal region','source.png',removal_box),
            ('Reconstructed source region','final.png',removal_box),
            ('Lifted bite and nearby utensil','final.png',target_box)]
    canvas=Image.new('RGB',(1260,480),'#edf1f5'); draw=ImageDraw.Draw(canvas)
    draw.text((12,8),row['case_id']+' / detail views; small native regions are enlarged',font=font(21),fill='#172b3f')
    boxes={}
    for i,(label,name,box) in enumerate(panels):
        draw.text((i*420+10,45),label,font=font(17),fill='#172b3f')
        if box:
            crop=Image.open(folder/name).convert('RGB').crop(box)
            scale=min(400/crop.width,360/crop.height)
            crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
            canvas.paste(crop,(i*420+(420-crop.width)//2,75+(360-crop.height)//2))
            draw.text((i*420+10,447),f'native crop: {box[2]-box[0]} x {box[3]-box[1]} pixels',font=font(16),fill='#496073')
            boxes[label]=list(box)
    path=out/'details'/f"{row['case_id']}.jpg"; path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=95)
    return path,boxes


def make_feature_sheet(row,out):
    match=row['identity_diagnostics'].get('local_feature_matching',{})
    pairs=match.get('pairs',[])
    if not pairs:
        return None
    folder=Path(row['folder']); source=Image.open(folder/'source.png').convert('RGB'); final=Image.open(folder/'final.png').convert('RGB')
    scale=min(1,600/source.width)
    size=(round(source.width*scale),round(source.height*scale))
    source=source.resize(size); final=final.resize(size)
    canvas=Image.new('RGB',(size[0]*2,size[1]+48),'#edf1f5'); canvas.paste(source,(0,48)); canvas.paste(final,(size[0],48))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,10),'Sparse descriptor matches (green: affine inlier; orange: other). Not semantic proof.',font=font(17),fill='#172b3f')
    for pair in pairs[:100]:
        p=[pair['source_xy'][0]*scale,pair['source_xy'][1]*scale+48]
        q=[pair['target_xy'][0]*scale+size[0],pair['target_xy'][1]*scale+48]
        color='#00e090' if pair['affine_inlier'] else '#ff9e2c'
        draw.line([tuple(p),tuple(q)],fill=color,width=1)
        for x,y in (p,q): draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
    path=out/'details'/f"{row['case_id']}_matches.jpg"; canvas.save(path,quality=95)
    return path


def build_candidate_gallery(root, variant, baseline_root, guide_root):
    """Review an unselected named variant without fabricating final-case state."""
    out=root/'evaluation'/f'{variant}_comparison'; out.mkdir(parents=True,exist_ok=True)
    baseline_root=baseline_root.resolve(); guide_root=guide_root.resolve()
    rows=[]; pending=[]
    for guide in sorted(guide_root.glob('real_*')):
        candidate=root/'real'/guide.name/'candidates'/variant
        required=[candidate/n for n in ('projected.png','unprojected.png','request.json','generation.json')]
        if not all(p.is_file() for p in required):
            pending.append(guide.name); continue
        request=json.loads((candidate/'request.json').read_text(encoding='utf-8'))
        source=guide/'source.png'; projected=candidate/'projected.png'; baseline=baseline_root/guide.name/'final.png'
        validation={name:(guide/name).is_file() and sha256(guide/name)==value for name,value in request.get('input_hashes',{}).items()}
        if not validation or not all(validation.values()):
            raise ValueError(f'{guide.name}: guide hash mismatch or missing request provenance: {validation}')
        src=np.asarray(Image.open(source).convert('RGB')); dst=np.asarray(Image.open(projected).convert('RGB'))
        mask=np.asarray(Image.open(guide/'inpaint_mask.png').convert('L'))>0
        if src.shape!=dst.shape or src.shape[:2]!=mask.shape:
            raise ValueError(f'{guide.name}: image/mask dimensions differ')
        change=int(np.count_nonzero(np.any(src!=dst,axis=-1)&~mask))
        review_path=candidate/'observation_sam3_v2/visual_observation_review.json'
        review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
        if review and review.get('input_final_sha256')!=sha256(projected): review=None
        paths={'source':source,'mld3':baseline,'projected':projected,'unprojected':candidate/'unprojected.png',
               'request':candidate/'request.json','generation':candidate/'generation.json','edit_mask':guide/'inpaint_mask.png'}
        boxes={'source':bbox_from_masks(guide,['source_removal_mask.png'],16),
               'target':bbox_from_masks(guide,['target_food_mask.png'],24)}
        canvas=Image.new('RGB',(1440,480),'#edf1f5'); draw=ImageDraw.Draw(canvas)
        panels=((source,boxes['source'],'Original source region'),(projected,boxes['source'],'Generated source region'),
                (projected,boxes['target'],'Generated lifted bite'))
        draw.text((12,8),f'{guide.name} / {variant}; enlarged native crops',font=font(23),fill='#172b3f')
        for col,(path,box,label) in enumerate(panels):
            draw.text((480*col+10,46),label,font=font(20),fill='#172b3f')
            if box:
                crop=Image.open(path).convert('RGB').crop(box); crop.thumbnail((460,350))
                scale=min(460/crop.width,350/crop.height)
                crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
                canvas.paste(crop,(480*col+(480-crop.width)//2,78+(350-crop.height)//2))
                draw.text((480*col+10,450),f'Native crop {box[2]-box[0]} x {box[3]-box[1]} px',font=font(17),fill='#496073')
        detail=out/f'{guide.name}_details.jpg'; canvas.save(detail,quality=96); paths['details']=detail
        rows.append({'case_id':guide.name,'paths':{k:str(v) for k,v in paths.items()},
                     'sha256':{k:sha256(v) for k,v in paths.items() if v.is_file()},'guide_hash_checks':validation,
                     'outside_edit_changed_pixels':change,'crop_boxes_xyxy':boxes,'fresh_review':review})
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<style>body{font:16px/1.5 system-ui;background:#edf1f5;color:#172b3f;margin:24px}main{max-width:1500px;margin:auto}section{background:white;padding:18px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{max-width:100%;height:auto}figcaption{font-weight:600}.note{background:#fff3d4;padding:12px}a{color:#175da3}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style><main>',
           f'<h1>{h(variant)}: {len(rows)} actual candidate comparisons</h1>',
           '<p class="note">These are experimental candidates, with no final selection or aggregate realism score. The named result uses material projection; the unprojected reconstruction and actual request remain linked. Exact exterior preservation checks locality only. Enlarged crops expose source seams and small lifted-food errors.</p>',
           f'<p>Guide inputs match the recorded request for every displayed candidate. Outside-edit unchanged: {sum(r["outside_edit_changed_pixels"]==0 for r in rows)}/{len(rows)}. Fresh previous detailed visual reviews: {sum(r["fresh_review"] is not None for r in rows)}/{len(rows)}.</p>',
           '<p><a href="manifest.json">Image hashes, provenance checks, crop coordinates and available visual reviews</a></p>']
    if (out/'contact_sheet_screening.json').exists():
        parts.append('<p><a href="contact_sheet_screening.json">Direct contact-sheet screening and enlarged-crop failure observations</a> (triage, not a complete per-criterion review of all cases)</p>')
    sheets=[]
    for start in range(0,len(rows),4):
        block=rows[start:start+4]; canvas=Image.new('RGB',(1500,50+len(block)*365),'#edf1f5'); draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(('Original source','Previous MLD3',variant+' projected')):
            draw.text((500*col+10,10),label,font=font(23),fill='#172b3f')
        for r,row in enumerate(block):
            for col,key in enumerate(('source','mld3','projected')):
                im=fit_image(Path(row['paths'][key]),(490,320))
                if im: canvas.paste(im,(500*col+(500-im.width)//2,48+365*r+(320-im.height)//2))
            draw.text((10,374+365*r),row['case_id'],font=font(18),fill='#172b3f')
        sheet=out/f'comparison_{start:02d}_{start+len(block)-1:02d}.jpg'; canvas.save(sheet,quality=96); sheets.append(sheet)
        parts.append(figure(sheet,f'Cases {start}–{start+len(block)-1}; click for larger contact sheet',out))
    for row in rows:
        paths={k:Path(v) for k,v in row['paths'].items()}
        parts.append(f'<section id="{h(row["case_id"])}"><h2>{h(row["case_id"])}</h2><div class="grid">')
        for key,label in (('source','Source — full resolution'),('mld3','MLD3 — full resolution'),('projected',variant+' projected — full resolution')):
            parts.append(figure(paths[key],label,out))
        parts.append('</div><p>Outside frozen edit mask: '+str(row['outside_edit_changed_pixels'])+' changed pixels.</p>')
        parts.append('<p>'+' · '.join(f'<a href="{h(href(paths[k],out))}">{h(k)}</a>' for k in ('unprojected','request','generation','edit_mask'))+'</p>')
        parts.append(figure(paths['details'],'Source removal and target details',out))
        review=row['fresh_review']
        parts.append('<details><summary>Hash-bound visual review</summary><pre>'+h(json.dumps(review,indent=2) if review else 'Not yet reviewed against all visual criteria. A contact sheet is not a substitute for a detailed review.')+'</pre></details></section>')
    parts.append('</main></html>'); (out/'gallery.html').write_text('\n'.join(parts),encoding='utf-8')
    manifest={'variant':variant,'scope':'Experimental candidate comparison; no final selection and no aggregate realism score',
              'rows':rows,'pending':pending,'sheets':{p.name:sha256(p) for p in sheets},'script_sha256':sha256(Path(__file__))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(out/'gallery.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',help='Build an unselected candidate comparison without reform_summary.json')
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--guide-root',type=Path)
    args=parser.parse_args(); root=args.root.resolve(); out=root/'evaluation'
    if args.variant:
        if not args.baseline_root or not args.guide_root: parser.error('--variant requires --baseline-root and --guide-root')
        build_candidate_gallery(root,args.variant,args.baseline_root,args.guide_root)
        return
    summary=json.loads((out/'reform_summary.json').read_text(encoding='utf-8'))
    rows=summary['cases']; phase_label=summary.get('phase_label','MLD4 reconstruction'); parts=['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MLD4 actual image review</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#eff2f6;color:#172b3f;margin:0}main{max-width:1560px;margin:auto;padding:24px}h1{font-size:30px}h2{font-size:23px}h3{font-size:18px}p{max-width:1200px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}section{background:white;border:1px solid #d5dfe9;border-radius:10px;padding:18px;margin:22px 0}figure{margin:0}img{display:block;max-width:100%;height:auto}figure img{width:100%}figcaption{font-weight:600;margin-top:6px}.note{padding:12px;background:#fff5d7;border-left:4px solid #bc8b12}.good{color:#246447}.bad{color:#a92424}.data,pre{font:13px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f8;padding:12px}a{color:#175da3}details{margin:15px 0}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #dae1e8;vertical-align:top}.missing{height:150px;background:#e3e8ef;display:grid;place-items:center}.links{display:flex;gap:16px;flex-wrap:wrap}.pill{display:inline-block;border-radius:5px;padding:4px 9px;background:#edf2f8;margin:3px}.wide{margin:12px 0}.wide img{max-width:1260px;width:100%}@media(max-width:850px){main{padding:10px}.grid,.grid.three{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main>
''', f'<h1>{h(phase_label)}: actual image review</h1>',
        f'<p>{len(rows)} complete actual outputs. This gallery separates the source, previous result, geometric guide and generated reconstruction. Candidate images and native detail views remain available for inspection.</p>',
        '<p class="note">A zero background difference checks edit locality only. Descriptor similarity does not prove ingredient identity. Guide silhouettes are intentions, and do not establish the final food or utensil topology. Photographic realism is assessed through explicit, hash-bound visual review; no composite realism score is assigned.</p>',
        f'<p>Background locality passes: {summary["counts"]["locality_pass"]}/{len(rows)}. Fresh complete visual reviews: {summary["counts"]["fresh_complete_visual_review"]}/{len(rows)}. Hash-bound final segmentation: {summary["counts"]["hash_bound_final_observer"]}/{len(rows)}.</p>',
        '<p class="links"><a href="reform_summary.json">Metrics and provenance</a><a href="reform_cases.csv">CSV</a><a href="visual_review_template.json">Unfilled review template</a><a href="visual_criteria_protocol.md">Review criteria</a></p>',
        '<details><summary>Required visual criteria</summary><table>']
    for criterion,description in summary['criteria'].items():
        parts.append(f'<tr><th>{h(criterion)}</th><td>{h(description)}</td></tr>')
    parts.append('</table></details>')
    if summary['pending'] or summary['errors']:
        parts.append('<p class="note">Incomplete and invalid outputs are retained below; they are not silently counted as successful.</p><pre>'+h(json.dumps({'pending':summary['pending'],'errors':summary['errors']},indent=2))+'</pre>')
    sheets=[]; crop_metadata={}
    for row in rows:
        folder=Path(row['folder']); baseline=Path(row['baseline_final_path']) if row['baseline_final_path'] else None
        guide=folder/'guide.png' if (folder/'guide.png').exists() else folder/'deformed.png'
        im=row['image']; review=row['visual_review']; ident=row['identity_diagnostics']
        parts.append(f'<section id="{h(row["case_id"])}"><h2>{h(row["case_id"])}</h2><div class="grid">')
        for path,label in ((folder/'source.png','Source'),(baseline,'Previous MLD3 result'),(guide,'Geometric guide'),(folder/'final.png',phase_label)):
            parts.append(figure(path,label,out))
        parts.append('</div>')
        locality='good' if im['outside_edit_changed_pixels']==0 else 'bad'
        parts.append(f'<p><span class="pill {locality}">Outside edit: {im["outside_edit_changed_pixels"]} changed pixels</span><span class="pill">Edit envelope: {100*im["declared_edit_fraction"]:.2f}%</span><span class="pill">Review: {h(review["status"])}</span></p>')
        if row['state'].get('semantic_target'):
            parts.append(f'<p>Declared food target: {h(row["state"]["semantic_target"])}</p>')
        parts.append(f'<p class="note">{h(review.get("notes") or "No complete fresh visual judgment is recorded. Inspect the native image and both detail regions; metrics cannot supply the missing judgment.")}</p>')
        detail,boxes=make_detail_sheet(row,out); crop_metadata[row['case_id']]=boxes; sheets.append(detail)
        parts.append('<div class="wide">'+figure(detail,'Actual source and bite details; dimensions are printed on the sheet',out)+'</div>')
        parts.append('<details><summary>Visual criterion judgments</summary><table><tr><th>Criterion</th><th>Status and evidence</th></tr>')
        for key in summary['criteria']:
            value=review.get('criteria',{}).get(key,'unreviewed')
            text=json.dumps(value,ensure_ascii=False) if isinstance(value,dict) else value
            parts.append(f'<tr><th>{h(key)}</th><td>{h(text)}</td></tr>')
        parts.append('</table></details>')
        parts.append('<details><summary>Editing envelope, guide masks and independent final masks</summary><div class="grid">')
        for name,label in (('edit_mask.png','Frozen editing envelope'),('source_removed_mask.png','Intended source removal'),('moved_food_mask.png','Intended carried-food silhouette'),('spoon_mask.png','Intended utensil silhouette')):
            parts.append(figure(folder/name,label,out))
        for name,label in (('observed_final_food_mask.png','Final food observer (verify hash/provenance below)'),('observed_final_utensil_mask.png','Final utensil observer (verify hash/provenance below)')):
            if (folder/name).exists(): parts.append(figure(folder/name,label,out))
        parts.append('</div><pre class="data">'+h(json.dumps(row['observed_final'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<details><summary>Appearance and sparse source-target matching diagnostics</summary>')
        parts.append(f'<p>Chromaticity distribution distance: {number(ident.get("chromaticity_histogram_js_distance"))}; local texture distance: {number(ident.get("local_binary_pattern_js_distance"))}. Target region: {h(ident["target_region_provenance"])}. These have no claimed pass threshold.</p>')
        feature_sheet=make_feature_sheet(row,out)
        if feature_sheet:
            sheets.append(feature_sheet); parts.append(figure(feature_sheet,'Actual sparse feature correspondences',out))
        ident_small={k:v for k,v in ident.items() if k!='local_feature_matching'}
        matches={k:v for k,v in ident.get('local_feature_matching',{}).items() if k!='pairs'}
        parts.append('<pre>'+h(json.dumps({'identity':ident_small,'matching':matches,'boundary':row['boundary_diagnostics']},ensure_ascii=False,indent=2))+'</pre></details>')
        if row['candidates']:
            parts.append('<details><summary>All named generation variants and their projection</summary>')
            for candidate in row['candidates']:
                cfolder=Path(candidate['path']).parent
                parts.append(f'<h3>{h(candidate["variant"])} · outside edit changes: {candidate["outside_edit_changed_pixels"]}</h3><div class="grid three">')
                for name,label in (('generated.png','Raw model output'),('candidate.png','Aligned candidate'),('projected.png','Candidate projected into envelope')):
                    parts.append(figure(cfolder/name,label,out))
                parts.append('</div>')
                if (cfolder/'request.json').exists(): parts.append(f'<a href="{h(href(cfolder/"request.json",out))}">Actual generation request</a>')
            parts.append('</details>')
        parts.append('<details><summary>Full measured contract, silhouette proxies and execution metadata</summary><pre>'+h(json.dumps({'contract':row['contract_checks'],'guide_geometry':row['guide_geometry'],'state':row['state'],'artifact_sha256':row['artifact_sha256']},ensure_ascii=False,indent=2))+'</pre></details></section>')
    for start in range(0,len(rows),4):
        block=rows[start:start+4]; canvas=Image.new('RGB',(1600,50+len(block)*330),'#edf1f5'); draw=ImageDraw.Draw(canvas)
        for c,label in enumerate(('Source','Previous MLD3','Geometric guide',phase_label)):
            draw.text((400*c+10,10),label,font=font(22),fill='#172b3f')
        for r,row in enumerate(block):
            folder=Path(row['folder']); baseline=Path(row['baseline_final_path']) if row['baseline_final_path'] else None
            guide=folder/'guide.png' if (folder/'guide.png').exists() else folder/'deformed.png'
            for c,path in enumerate((folder/'source.png',baseline,guide,folder/'final.png')):
                im=fit_image(path,(390,285))
                if im: canvas.paste(im,(400*c+(400-im.width)//2,50+330*r+(285-im.height)//2))
            draw.text((10,340+r*330),row['case_id']+' / '+row['visual_review']['status'],font=font(17),fill='#172b3f')
        path=out/f'comparison_{start:02d}_{start+len(block)-1:02d}.jpg'; canvas.save(path,quality=94); sheets.append(path)
    parts.append('<p>All data are from an inspected development set. Generated hidden material is a plausible reconstruction, not recovered physical truth. Failures and uncertain judgments remain visible.</p></main></html>')
    (out/'gallery.html').write_text('\n'.join(parts),encoding='utf-8')
    manifest={'case_count':len(rows),'gallery':str(out/'gallery.html'),
              'input_summary_sha256':sha256(out/'reform_summary.json'),
              'sheets':{href(path,out):sha256(path) for path in sheets},'detail_crop_boxes_xyxy':crop_metadata,
              'scope':'Actual output comparisons. No automatic realism rating. Resized detail sheets supplement native files.'}
    (out/'gallery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out/'gallery.html')


if __name__=='__main__':
    main()
