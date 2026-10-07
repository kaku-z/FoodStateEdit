import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_first_bite_ballots import aggregate,joint_success,DIMENSIONS


class FirstBiteBallotTests(unittest.TestCase):
    def test_empty_and_uncertain_cannot_pass(self):
        with self.assertRaises(ValueError):joint_success({})
        row={k:'pass' for k in DIMENSIONS};row.update(photo_score='4',lift='uncertain')
        self.assertFalse(joint_success(row))

    def test_disjoint_rater_failures_do_not_create_a_false_joint_pass(self):
        keys=[];ballots=[]
        for i,failed in enumerate(['lift','source_change','hand_and_utensil']):
            package=f'p{i}';bid=f'b{i}'
            keys.append({'package':package,'blind_id':bid,'run_id':'case__method__281'})
            row={k:'pass' for k in DIMENSIONS};row.update(photo_score='5');row[failed]='fail'
            ballots.append({'rater':f'actual-test-placeholder-{i}','package':package,'ratings':{bid:row}})
        self.assertFalse(aggregate(keys,ballots)[0]['joint_pass'])
        ballots[2]['rater']=ballots[0]['rater']
        with self.assertRaises(ValueError):aggregate(keys,ballots)


if __name__=='__main__':unittest.main()
