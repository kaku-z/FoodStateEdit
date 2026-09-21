"""Copy the frozen DiffSynth training runtime and add the opt-in SCFST hook."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_scfst_vace_runtime import REF_OLD, REF_NEW, BLOCK_OLD, BLOCK_NEW


VIDEO_OLD = '''    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        num_frames = self.get_num_frames(reader)
        total_raw_frames = reader.count_frames()
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames
'''


VIDEO_NEW = '''    def __call__(self, data: str):
        # imageio's PyAV legacy reader can fail in get_meta_data() when it
        # seeks with index=None (imageio 2.37 / PyAV 14). Decode through PyAV
        # directly so the private training runtime does not depend on that
        # broken metadata path. This keeps the original frame-sampling rules.
        import av
        with av.open(data) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            raw_frame_rate = float(rate) if rate is not None else float(self.frame_rate)
            raw_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        total_raw_frames = len(raw_frames)
        if total_raw_frames < 1:
            raise ValueError(f"Video contains no decodable frames: {data}")
        if self.fix_frame_rate:
            duration = total_raw_frames / raw_frame_rate
            total_available_frames = math.floor(duration * self.frame_rate)
        else:
            total_available_frames = total_raw_frames
        num_frames = min(self.num_frames, total_available_frames)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        frames = []
        for frame_id in range(num_frames):
            source_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = Image.fromarray(raw_frames[source_id])
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames
'''


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-runtime',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args(); source=a.source_runtime.resolve(); target=a.bundle.resolve()/'runtime'
    if target.exists(): raise FileExistsError(target)
    rel=Path('diffsynth/pipelines/wan_video.py')
    operators_rel=Path('diffsynth/core/data/operators.py')
    code=(source/rel).read_text(encoding='utf-8')
    operators_code=(source/operators_rel).read_text(encoding='utf-8')
    for old,name in ((REF_OLD,'reference'),(BLOCK_OLD,'block')):
        if code.count(old)!=1: raise ValueError(f'expected one {name} site, found {code.count(old)}')
    if operators_code.count(VIDEO_OLD)!=1:
        raise ValueError(f'expected one LoadVideo site, found {operators_code.count(VIDEO_OLD)}')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','.git'))
    patched=code.replace(REF_OLD,REF_NEW).replace(BLOCK_OLD,BLOCK_NEW)
    patched_operators=operators_code.replace(VIDEO_OLD,VIDEO_NEW)
    compile(patched,str(target/rel),'exec')
    compile(patched_operators,str(target/operators_rel),'exec')
    (target/rel).write_text(patched,encoding='utf-8')
    (target/operators_rel).write_text(patched_operators,encoding='utf-8')
    receipt={
        'schema_version':'foodstateedit.scfst_training_runtime.v2',
        'source_runtime':str(source),'source_pipeline_sha256':sha256(source/rel),
        'patched_pipeline_sha256':sha256(target/rel),'preparer_sha256':sha256(Path(__file__)),
        'source_operators_sha256':sha256(source/operators_rel),
        'patched_operators_sha256':sha256(target/operators_rel),
        'video_loader':'direct PyAV decode preserving the original frame-sampling rules',
        'model_behavior':'unchanged unless state_transfer_adapter is attached',
        'data_behavior':'same sampling policy, with direct PyAV decoding instead of imageio legacy metadata',
    }
    (a.bundle/'runtime_derivation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__': main()
