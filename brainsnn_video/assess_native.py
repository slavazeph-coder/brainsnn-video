"""Frozen native-resolution postprocess test; no learned-model quality claim.

Creates identically encoded A/B clips and a private random mapping for a human
reviewer. The numerical operator is the existing unmodified ENG-1 engine.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import time

import numpy as np
import torch

LEGACY_ROOT = Path(__file__).resolve().parent / 'engine'
SCRIPT = LEGACY_ROOT/'crumb_coherence/scripts/run_eng1_1080.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def gates(detail, motion, jitter, wall):
    return {'detail': detail >= .97, 'motion': .9 <= motion <= 1.1,
            'jitter': jitter <= 1.5, 'runtime': wall <= 1800}


def validate_cadence(timestamps,fps,duration,source_role):
    if source_role=='authorized-client' and duration>15:
        raise ValueError('Frozen offer refuses client clips longer than 15 seconds')
    if len(timestamps)<2 or fps<=0:
        raise ValueError('A finite moving clip with known frame cadence is required')
    step=1/fps; tolerance=max(1e-4,step*.001)
    if any(not np.isfinite(t) for t in timestamps) or any(abs((b-a)-step)>tolerance for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('VFR/discontinuous timestamps refused; this encoder must not retime footage silently')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-role', choices=['owned-generated', 'authorized-client'], required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh result identity required')
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration','-of','json',str(args.input)]))
    stream=metadata['streams'][0]
    width,height=int(stream['width']),int(stream['height'])
    if (width,height) not in ((1280,720),(720,1280),(1920,1080),(1080,1920)):
        raise ValueError('This native gate refuses resizing or lower-resolution inputs')
    n,d=stream['r_frame_rate'].split('/'); fps=float(n)/float(d)
    expected=int(stream['nb_frames'])
    frame_info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(args.input)]))
    timestamps=[float(f['best_effort_timestamp_time']) for f in frame_info['frames']]
    validate_cadence(timestamps,fps,float(metadata['format']['duration']),args.source_role)
    if len(timestamps)!=expected: raise ValueError('Frame count disagrees with actual timestamp observations')
    sys.path.insert(0,str(LEGACY_ROOT))
    spec=importlib.util.spec_from_file_location('frozen_eng1',SCRIPT)
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    files=[SCRIPT,LEGACY_ROOT/'crumb_coherence/core.py',LEGACY_ROOT/'crumb_coherence/metrics.py',Path(__file__)]
    sources={str(p):sha(p) for p in files}
    args.out.mkdir(parents=True,mode=0o700)
    private=args.out/'private'; private.mkdir(mode=0o700)
    mapping={'A': 'candidate' if secrets.randbits(1) else 'pass_through'}
    mapping['B']='pass_through' if mapping['A']=='candidate' else 'candidate'
    key={'mapping':mapping,'human_review_status':'not_received','do_not_share_with_reviewer':True}
    save(private/'blind-key.json',key); (private/'blind-key.json').chmod(0o600)
    save(args.out/'preregistration.json',{'source_sha256':sources,'input_sha256':sha(args.input),
        'source_role':args.source_role,'independent_stranger_clip':False,
        'creator_relationship_verification':'not established by permission/source role',
        'dimensions':[width,height],'fps':fps,'expected_frames':expected,
        'operator_config':legacy.SHIP_CFG,'baseline':'identically encoded pass-through input',
        'additional_baseline':'frozen first-frame motion=0, a deliberately bad non-interference control',
        'gates':{'detail':.97,'motion':[.9,1.1],'jitter_max_over_p95':1.5,'wall_seconds':1800},
        'trained_weights':False,'gpu_used':False,'human_review_required':True,
        'claimed_temporal_memory_or_generation_quality':False})
    torch.set_num_threads(1)
    engine=legacy.SpectralCoherenceEngine(**legacy.SHIP_CFG); state=engine.init_state(height,width)
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(args.input),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    encoders={label:legacy.encoder_pipe(width,height,fps,str(args.out/(label+'.mp4'))) for label in ('A','B')}
    detail=[]; raw_diff=[]; cor_diff=[]; frame_metrics=[]; prev_raw=prev_cor=None
    started=time.monotonic(); count=0
    try:
        while True:
            buf=legacy._read_exact(decoder.stdout,width*height*3)
            if buf is None: break
            raw=torch.from_numpy(np.frombuffer(buf,np.uint8).reshape(height,width,3).copy()).float().permute(2,0,1).contiguous()/255.
            cor,state=engine.process_frame(raw,state)
            if not bool(torch.isfinite(cor).all()): raise RuntimeError('Nonfinite corrected frame')
            detail.append(legacy.pearson(legacy.hf_luma_vec(raw),legacy.hf_luma_vec(cor)))
            if prev_raw is not None:
                raw_diff.append(float((raw-prev_raw).abs().mean())); cor_diff.append(float((cor-prev_cor).abs().mean()))
            frame_metrics.append({'frame':count,'detail_correlation':detail[-1],
                'raw_motion_mae':raw_diff[-1] if count else None,'candidate_motion_mae':cor_diff[-1] if count else None})
            for label,role in mapping.items():
                frame=cor if role=='candidate' else raw
                arr=(frame.clamp(0,1).permute(1,2,0)*255).round().to(torch.uint8).contiguous().numpy()
                encoders[label].stdin.write(arr.tobytes())
            prev_raw,prev_cor=raw,cor; count+=1
            if time.monotonic()-started>1800: raise TimeoutError('Native gate exceeded frozen runtime budget')
    finally:
        decoder.stdout.close()
        decoder_rc=decoder.wait(timeout=20)
        statuses=[]
        for enc in encoders.values():
            enc.stdin.close(); statuses.append(enc.wait(timeout=20))
    wall=time.monotonic()-started
    if decoder_rc or any(statuses) or count!=expected: raise RuntimeError('Incomplete decode/encode; do not accept this test')
    rd,cd=torch.tensor(raw_diff),torch.tensor(cor_diff)
    motion=float(cd.sum()/(rd.sum()+1e-8))
    jitter=float(cd.max()/(cd.quantile(.95)+1e-8))
    med=float(torch.tensor(detail).median())
    checks=gates(med,motion,jitter,wall)
    unchanged=all(sha(Path(p))==h for p,h in sources.items())
    if not unchanged: raise RuntimeError('Source changed during test')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'frames':count,'dimensions':[width,height],'fps':fps,'detail_median':med,
        'motion_ratio':motion,'jitter_candidate_max_over_p95':jitter,
        'jitter_input_max_over_p95':float(rd.max()/(rd.quantile(.95)+1e-8)),
        'wall_seconds':wall,'peak_process_rss_bytes':peak if sys.platform=='darwin' else peak*1024,
        'hardware':platform.platform(),'torch':torch.__version__,'threads':1,
        'trained_weights':False,'operator':'frozen analytic ENG-1 postprocess',
        'source_role':args.source_role,'independent_stranger_clips_verified':0,
        'creator_relationship_verification':'Requires separate identity and relationship evidence',
        'source_unchanged':unchanged,'output_sha256':{k:sha(args.out/(k+'.mp4')) for k in mapping},
        'blind_mapping_shared':False,'human_review':'pending','first_collapse':'not_applicable_to_analytic_nonautoregressive_postprocess',
        'paid_orders':0,'commercial_success_claim_allowed':False,'video_generation_quality_claim_allowed':False}
    save(args.out/'receipt.json',receipt)
    with (args.out/'frame-metrics.jsonl').open('w') as f:
        for row in frame_metrics: f.write(json.dumps(row)+'\n')
    (args.out/'REVIEW.md').write_text('# Native clip review\n\nWatch A.mp4 and B.mp4 in full at native resolution. Do not inspect the private key. Record whether either clip better preserves intended motion, lighting, detail and object identity, and whether either introduces flicker, blur, warping or ghost objects. Say neither when neither is better. Numerical gates and this owned generated clip do not establish customer demand or an independent human result.\n')
    print(json.dumps({k:receipt[k] for k in ('status','checks','frames','detail_median','motion_ratio','wall_seconds')}))


if __name__=='__main__': main()
