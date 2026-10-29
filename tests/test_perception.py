from dataclasses import replace
from types import SimpleNamespace
import pytest
from cortex.perception import (Detection, NameCandidate, PerceptionFrame, RelationProposal, PerceptionAdapter)

class Runtime:
    def __init__(self): self.calls=[]
    def step(self,*args):
        self.calls.append(args)
        return SimpleNamespace(bindings=tuple(10 if row[0] > 0 else 20 for row in args[0]),unresolved=False)

def frame(i=0,names=('Paris','Paris')):
    return PerceptionFrame(str(i),float(i),'clip','encoder:v1','space:v1',(
        Detection('a',(1.,0.),(NameCandidate(names[0],.9,kind='name'),)),
        Detection('b',(-1.,0.),(NameCandidate(names[1],.9,kind='name'),))),
        (RelationProposal('r','a','left-of','b',.8),))

def test_names_do_not_merge_entities_and_relations_preserve_roles():
    runtime=Runtime();adapter=PerceptionAdapter(runtime,feature_space='space:v1',feature_dim=2)
    read=adapter.observe(frame())
    assert read.bindings == (('a',10),('b',20))
    assert adapter.label(10) == 'Paris [entity:10]'
    assert adapter.label(20) == 'Paris [entity:20]'
    assert runtime.calls[0][1] == []
    assert adapter.accepted_statements('0',[]) == ()
    statement=adapter.accepted_statements('0',['r'])[0]
    assert (statement.subject,statement.property,statement.value)==('entity:10','left-of','entity:20')
    assert statement.references == ('clip',)

def test_reorder_labels_and_conflicts_do_not_change_binding():
    adapter=PerceptionAdapter(Runtime(),feature_space='space:v1',feature_dim=2)
    adapter.observe(frame())
    second=frame(1,('London','Paris'))
    result=adapter.observe(replace(second,detections=tuple(reversed(second.detections))))
    assert result.bindings == (('b',20),('a',10))
    assert adapter.label(10)=='entity:10'
    assert len(adapter.names)==4

def test_invalid_inputs_fail_before_runtime_and_empty_frames_are_recorded():
    runtime=Runtime();adapter=PerceptionAdapter(runtime,feature_space='space:v1',feature_dim=2,max_frames=2)
    for f in [replace(frame(),feature_space='other'),replace(frame(),detections=(Detection('a',(1.,)),),relations=())]:
        with pytest.raises(ValueError): adapter.observe(f)
    assert not runtime.calls
    adapter.observe(replace(frame(),detections=(),relations=()))
    assert not runtime.calls
    with pytest.raises(ValueError): adapter.observe(frame())
    adapter.observe(frame(1))
    with pytest.raises(RuntimeError): adapter.observe(frame(2))
    with pytest.raises(ValueError): replace(frame(),relations=(RelationProposal('x','missing','near','a',.9),))

def test_invalid_runtime_response_poisoned_and_unresolved_names_not_adopted():
    runtime=Runtime();runtime.step=lambda *a: SimpleNamespace(bindings=(1,1))
    adapter=PerceptionAdapter(runtime,feature_space='space:v1',feature_dim=2)
    with pytest.raises(ValueError):adapter.observe(frame())
    assert not adapter.names and not adapter.reads
    with pytest.raises(RuntimeError):adapter.observe(frame(1))
    runtime.step=lambda *a: SimpleNamespace(bindings=(1,2),unresolved=True)
    adapter=PerceptionAdapter(runtime,feature_space='space:v1',feature_dim=2)
    assert adapter.observe(frame()).unresolved
    assert not adapter.names
    with pytest.raises(ValueError):adapter.accepted_statements('0',['r'])

@pytest.mark.parametrize('score',[float('nan'),float('inf'),-1,2])
def test_bad_confidence_rejected(score):
    with pytest.raises(ValueError):NameCandidate('cat',score)


def test_generated_video_with_reference_binding():
    import shutil
    import sys
    from pathlib import Path
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is an optional video dependency')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
    from perception_video_demo import run
    report=run()
    assert report['decoded_frames']==12
    assert report['identity_switches']==0
    assert report['persistent_entities']==2
    assert report['relation_directions_observed']==2
    assert not report['neural_model_used']
