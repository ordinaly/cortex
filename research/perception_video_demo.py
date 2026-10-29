"""Real video decode / reference-core integration fixture, NOT neural perception."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'/'v0_9_5'))
from cortex_v07 import DependencyIntegratedRuntime, Frame, Detection as ReferenceDetection
from cortex.perception import Detection, NameCandidate, PerceptionFrame, RelationProposal, PerceptionAdapter
from cortex.video_perception import ingest_video


class ReferenceRuntime:
    def __init__(self):
        self.core=DependencyIntegratedRuntime(feature_dim=3,max_entities=16,group_candidates=[(0,)])
        # Identity-only geometry: rotating RGB coordinates would change color.
        self.core.state.all_shifts=(0,)
        self.core.state.subgroup=(0,)
    def step(self,detections,relation_obs,intervention_src_det,outcomes,outcome):
        return self.core.step(Frame(tuple(ReferenceDetection(np.asarray(x)) for x in detections),
                                    tuple(relation_obs),intervention_src_det,tuple(outcomes)))


class ColorFixtureProvider:
    """Known-color segmentation, intentionally not advertised as a neural model."""
    def perceive(self,image,*,frame_id,timestamp,source):
        detections=[];centers={}
        for channel,name in [(0,'red object'),(1,'green object')]:
            mask=(image[:,:,channel]>180)&(image[:,:,2]<60)
            ys,xs=np.where(mask)
            if not len(xs):continue
            key=str(channel)
            # Features come from pixels; neither name nor ground-truth track ID is fed to binding.
            features=tuple(float(x) for x in image[mask].mean(axis=0)/127.5)
            detections.append(Detection(key,features,(NameCandidate(name,.95),)))
            centers[key]=float(xs.mean())
        if int(frame_id.split(':')[-1])%2:
            detections.reverse()
        relations=[]
        if len(centers)==2 and centers['0']!=centers['1']:
            left,right=sorted(centers,key=centers.get)
            relations=[RelationProposal('horizontal',left,'left-of',right,1.0)]
        return PerceptionFrame(frame_id,timestamp,source,'color-fixture:v1','rgb-mean:v1',
                               tuple(detections),tuple(relations))


def make_clip(path):
    frames=[]
    for i in range(12):
        image=np.zeros((96,128,3),dtype=np.uint8)
        if i!=6:
            x=8+8*i
            image[15:29,x:x+12,0]=255
            if i not in (4,5):
                x=108-8*i
                image[60:74,x:x+12,1]=255
        frames.append(image)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                    '-s','128x96','-r','4','-i','pipe:0','-c:v','ffv1',str(path)],
                   input=b''.join(f.tobytes() for f in frames),check=True,timeout=30)


def run():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'controlled.mkv'
        make_clip(path)
        adapter=PerceptionAdapter(ReferenceRuntime(),feature_space='rgb-mean:v1',feature_dim=3)
        reads=ingest_video(path,ColorFixtureProvider(),adapter,sample_fps=4,max_frames=12,width=128,height=96)
        tracks={key:{dict(r.bindings)[key] for r in reads if key in dict(r.bindings)} for key in ('0','1')}
        assert all(len(ids)==1 for ids in tracks.values()) and tracks['0']!=tracks['1']
        assert len(reads)==12 and not reads[6].bindings
        directions={(r.subject_id,r.object_id) for read in reads for r in read.relations}
        assert len(directions)==2
        return {'fixture':'generated lossless video; classical color segmentation; frozen Python reference core',
                'neural_model_used':False,'native_extension_used':False,
                'decoded_frames':len(reads),'empty_frames':sum(not r.bindings for r in reads),
                'persistent_entities':len(set.union(*tracks.values())),
                'identity_switches':sum(len(ids)-1 for ids in tracks.values()),
                'relation_directions_observed':len(directions),
                'labels':{str(i):adapter.label(i) for i in sorted(set.union(*tracks.values()))}}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
