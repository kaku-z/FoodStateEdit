"""Verify and display all height diagnostic outputs; do not score automatically."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];b=root/'outputs/first_bite_20260928_height_bundle_v1'
    c=json.loads((b/'config.json').read_text());found={};manifests=[]
    for folder in sorted((root/'outputs/first_bite_20260928_height_gp40').glob('height_shard_*_v1')):
        mp=folder/'run_manifest.json';m=json.loads(mp.read_text())
        assert m['status']=='complete_unreviewed' and not m['failed']
        assert m['config_sha256']==sha(b/'config.json')
        manifests.append({'path':str(mp),'sha256':sha(mp),'run':m})
        for row in m['completed']:
            assert row['name'] not in found
            for name,h in row['files'].items():assert sha(folder/row['name']/name)==h
            case=next(x for x in c['cases'] if x['case_id']==row['case_id'])
            assert hashlib.sha256(case['prompts']['D_geometry_guide'].encode()).hexdigest()==row['prompt_sha256']
            assert row['seed']==2027 and row['raw_size']==case['output_size']
            found[row['name']]={'record':row,'raw':folder/row['name']/'raw.png'}
    expected={x['case_id']+'__D_geometry_guide__2027' for x in c['cases']}
    assert set(found)==expected and len(found)==12
    for case in c['cases']:
        for f in case['files'].values():assert sha(b/f['path'])==f['sha256']
    a.output.mkdir(parents=True,exist_ok=False)
    for family in ['cohesive','granular','strand','liquid']:
        cases=[x for x in c['cases'] if x['family']==family]
        assert len({x['files']['source.png']['sha256'] for x in cases})==1
        assert len({x['prompts']['D_geometry_guide'] for x in cases})==1
        cells=[('SOURCE',b/cases[0]['files']['source.png']['path'])]
        cells += [(f'h={case["diagnostic_height"]} seed=2027',found[case['case_id']+'__D_geometry_guide__2027']['raw']) for case in cases]
        canvas=Image.new('RGB',(1600,1280),'#edf1f5');d=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate(cells):
            im=Image.open(path).convert('RGB');im.thumbnail((800,600),Image.Resampling.LANCZOS)
            x=i%2*800;y=i//2*640;canvas.paste(im,(x+(800-im.width)//2,y+(600-im.height)//2))
            d.text((x+10,y+614),label,fill='black')
        canvas.save(a.output/(family+'_height_all.png'))
    data={'status':'12_HEIGHT_OUTPUTS_HASH_VERIFIED_INTERPRETATION_REQUIRED','calls':12,'main_denominator_includes_these':False,
          'identical_prompts_and_sources_per_family':True,'manifests':manifests,'independent_ballots_received':0}
    (a.output/'verification.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'verified':12,'grids':4,'interpretation':'pending'}))


if __name__=='__main__':main()
