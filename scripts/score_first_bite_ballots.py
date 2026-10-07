"""Import actual completed reviewer exports; never synthesize missing votes."""
import argparse,collections,json
from pathlib import Path

DIMENSIONS=('lift','bite_and_support','source_change','food_and_scene','hand_and_utensil')


def joint_success(row):
    for key in DIMENSIONS:
        if row.get(key) not in ['pass','fail','uncertain']:
            raise ValueError('Incomplete or invalid rating: '+key)
    score=row.get('photo_score')
    if str(score) not in ['1','2','3','4','5']:
        raise ValueError('Incomplete or invalid photo score')
    return all(row[key]=='pass' for key in DIMENSIONS) and int(score)>=4


def aggregate(keys,ballots):
    if len(ballots)!=3:raise ValueError('Exactly three completed reviewer exports required')
    identity=[x.get('rater','').strip().casefold() for x in ballots]
    if not all(identity) or len(set(identity))!=3:raise ValueError('Three distinct nonempty reviewer IDs required')
    lookup=collections.defaultdict(dict)
    for key in keys:
        if key['blind_id'] in lookup[key['package']]:raise ValueError('Duplicate review mapping')
        lookup[key['package']][key['blind_id']]=key['run_id']
    if len({x['package'] for x in ballots})!=3:raise ValueError('Duplicate package')
    votes=collections.defaultdict(list)
    for ballot in ballots:
        mapping=lookup.get(ballot['package'])
        if not mapping or set(ballot['ratings'])!=set(mapping):raise ValueError('Missing or extra ballot items')
        for bid,row in ballot['ratings'].items():
            votes[mapping[bid]].append({'rater':ballot['rater'],'joint_pass':joint_success(row),'rating':row})
    if any(len(v)!=3 for v in votes.values()):raise ValueError('Incomplete per-image panel')
    return [{'run_id':run,'joint_pass':sum(v['joint_pass'] for v in panel)>=2,'votes':panel} for run,panel in sorted(votes.items())]


def main():
    p=argparse.ArgumentParser();p.add_argument('--keys',type=Path,required=True);p.add_argument('--ballots',nargs=3,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=aggregate(json.loads(a.keys.read_text()),[json.loads(x.read_text()) for x in a.ballots])
    if a.output.exists():raise FileExistsError(a.output)
    by_method=collections.defaultdict(lambda:{'pass':0,'total':0})
    for row in results:
        method=row['run_id'].split('__')[1];by_method[method]['total']+=1;by_method[method]['pass']+=int(row['joint_pass'])
    out={'status':'COMPLETED_BALLOTS_IMPORTED','independence_and_human_origin':'Must be verified from actual reviewer provenance, not inferred from three JSON files.',
         'by_method':dict(by_method),'results':results}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(by_method)))


if __name__=='__main__':main()
