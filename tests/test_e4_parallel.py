"""并行E4的调度、恢复、选择一致性与真实短训练检查。"""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch,Mock
from dual_whisker_rl.arrival_protocol import atomic_json,digest,read,sha,model_header
from scripts.e4_task_queue import Queue,cached,process_from_receipt,room_for_model
from scripts.e4_parallel import aggregate_b1,aggregate_b2,select_pairs,make_job,probe_case,read_result

ROOT=Path(__file__).resolve().parents[1]
FAKE=r'''import sys,json,time,hashlib,os
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text());base=Path(j['events']);base.mkdir(exist_ok=True)
with (base/(j['id']+'.once')).open('x') as f:f.write('once')
(base/(j['id']+'.start')).write_text(str(time.time()))
time.sleep(j.get('delay',.15))
if j.get('fail'):sys.exit(7)
d={'job_signature':j['signature'],'kind':j['kind'],'result':{'id':j['id']},'artifacts':[]}
d['result_sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
p=Path(j['output']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
(base/(j['id']+'.end')).write_text(str(time.time()))
'''


class ParallelQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.worker=self.root/'worker.py';self.worker.write_text(FAKE,encoding='utf8')
        self.q=Queue(self.root/'queue',self.worker,self.root,interval=.01)
    def tearDown(self):self.tmp.cleanup()
    def job(self,id,family='G',**kw):
        return dict(id=id,signature=digest(id),kind='b1',family=family,events=str(self.root/'events'),
                    output=str(self.root/'results'/f'{id}.json'),**kw)

    def test_no_duplicate_dispatch_at_most_two_and_full_cache_skip(self):
        jobs=[self.job(str(i),delay=.25) for i in range(5)]
        self.q.run(jobs,limit=2,resource_guard=False)
        events=[]
        for job in jobs:
            events += [(float((self.root/'events'/f'{job["id"]}.start').read_text()),1),
                       (float((self.root/'events'/f'{job["id"]}.end').read_text()),-1)]
        count=peak=0
        for _,delta in sorted(events):count+=delta;peak=max(peak,count)
        self.assertEqual(peak,2)
        with patch.object(self.q,'start',side_effect=AssertionError('缓存不能重跑')):
            self.assertEqual(self.q.run(jobs,limit=2,resource_guard=False)['reused'],5)

    def test_incompatible_pair_waits_and_skips_to_compatible_task(self):
        jobs=[self.job('g0',delay=.2),self.job('m0','M',delay=.1),self.job('g1',delay=.2)]
        self.q.run(jobs,limit=2,allowed={'GG':True,'GM':False,'MM':False},resource_guard=False)
        p=self.root/'events'
        self.assertLess(float((p/'g1.start').read_text()),float((p/'g0.end').read_text()))
        self.assertGreaterEqual(float((p/'m0.start').read_text()),max(float((p/'g0.end').read_text()),float((p/'g1.end').read_text())))

    def test_failure_stops_new_dispatch_but_finishes_healthy_task(self):
        jobs=[self.job('bad',delay=.03,fail=True),self.job('healthy',delay=.25),self.job('pending')]
        with self.assertRaises(RuntimeError):self.q.run(jobs,limit=2,resource_guard=False)
        self.assertTrue(cached(jobs[1]));self.assertFalse((self.root/'events/pending.once').exists())
        self.assertEqual(read(self.q.receipt_path(jobs[0]))['status'],'failed')
        with patch.object(self.q,'start',side_effect=AssertionError('失败任务不能自动重试')):
            with self.assertRaises(RuntimeError):self.q.run(jobs,limit=2,resource_guard=False)

    def test_surviving_task_is_attached_without_second_process(self):
        job=self.job('survivor',delay=.5);old=self.q.start(job)
        old['receipt']['status']='monitor_interrupted';atomic_json(self.q.receipt_path(job),old['receipt'])
        try:
            with patch.object(self.q,'start',side_effect=AssertionError('存活任务不能重启')):
                result=self.q.run([job],limit=2,resource_guard=False)
            self.assertEqual(result['attached'],1);self.assertEqual(result['executed'],0)
        finally:old['popen'].wait();old['handle'].close()

    def test_pid_reuse_not_attached_or_signalled(self):
        p=Mock();p.create_time.return_value=20.
        with patch('scripts.e4_task_queue.psutil.Process',return_value=p):
            self.assertIsNone(process_from_receipt(dict(pid=100,created=10.,command=['old'])))
        p.kill.assert_not_called();p.cmdline.assert_not_called()

    def test_empty_command_keeps_slot_until_process_finishes(self):
        p=Mock();p.create_time.return_value=10.;p.status.return_value='running';p.cmdline.return_value=[]
        with patch('scripts.e4_task_queue.psutil.Process',return_value=p):
            self.assertIs(process_from_receipt(dict(pid=100,created=10.,command=['old'])),p)
        p.cmdline.return_value=['different']
        with patch('scripts.e4_task_queue.psutil.Process',return_value=p):
            with self.assertRaises(RuntimeError):process_from_receipt(dict(pid=100,created=10.,command=['old']))
    def test_exit_between_status_and_command_is_finished(self):
        import psutil
        p=Mock();p.create_time.return_value=10.;p.status.side_effect=['running',psutil.STATUS_ZOMBIE]
        p.cmdline.return_value=[]
        with patch('scripts.e4_task_queue.psutil.Process',return_value=p):
            self.assertIsNone(process_from_receipt(dict(pid=100,created=10.,command=['old'])))

    def test_reservations_prevent_startup_memory_overcommit(self):
        snapshot=dict(memory_limit=30*2**30,memory_current=20*2**30,gpu=[])
        with patch('scripts.e4_task_queue.resources',side_effect=lambda:copy.deepcopy(snapshot)):
            self.q.run([self.job('r0',delay=.2),self.job('r1',delay=.1)],limit=2,reserve={'memory':3*2**30})
        p=self.root/'events'
        self.assertGreaterEqual(float((p/'r1.start').read_text()),float((p/'r0.end').read_text()))

    def test_resource_margin_and_changed_source(self):
        self.assertFalse(room_for_model(dict(memory_limit=30*2**30,memory_current=29*2**30,gpu=[])))
        self.assertFalse(room_for_model(dict(memory_limit=30*2**30,memory_current=0,gpu=[{'free':2**30}])))
        self.assertFalse(room_for_model(dict(memory_limit=30*2**30,memory_current=0,gpu=[]),{'uses_gpu':True}))
        job=self.job('cached');self.q.run([job],resource_guard=False)
        changed=dict(job,signature='changed')
        with self.assertRaises(ValueError):cached(changed)
        d=read(job['output']);d['result']['id']='tampered';atomic_json(job['output'],d)
        self.assertFalse(cached(job))


class ParallelCalibrationTests(unittest.TestCase):
    def test_pair_selection_uses_total_throughput_and_harmonic_reference(self):
        rows=[dict(case=c,valid=True,steps_per_second=s) for c,s in [('G',100),('M',50),('GG',120),('MM',54),('GM',75)]]
        result=select_pairs(rows)
        self.assertEqual(result['allowed'],{'GG':True,'MM':False,'GM':True})
        rows[2]['valid']=False;self.assertFalse(select_pairs(rows)['allowed']['GG'])
        rows[0]['valid']=False
        with self.assertRaises(RuntimeError):select_pairs(rows)

    def test_parallel_calibration_matches_original_serial_full_payload(self):
        from scripts import run_e4_experiments as original
        cfgs=original.b1_calibration_grid()[:2]
        class Controller:
            def __init__(self,**kwargs):self.hit=kwargs['hit_threshold']
            def reset(self):pass
        def score(sector,seed):return dict(success_rate=sector/10,mean_final_distance=1.-sector/20)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(profile='formal',dry_run=False,resume=True,seeds=[1,2],max_steps=8)
            def train(args,root,method,seed,*,fixed_sector,task_dir_override):
                p=Path(task_dir_override)/'checkpoints';p.mkdir(parents=True,exist_ok=True)
                (p/f'b2_fixed_seed{seed}_102400_steps.zip').touch()
            with patch.object(original,'calibration_signature',return_value='sig'), \
                 patch.object(original,'legacy_calibration_signatures',return_value=set()), \
                 patch.object(original,'load_completed_calibration',return_value=None), \
                 patch.object(original,'b1_calibration_grid',return_value=cfgs), \
                 patch.object(original,'base_config',return_value={}), \
                 patch.object(original,'read_scenarios',return_value=[1,2]), \
                 patch.object(original,'make_env',return_value=SimpleNamespace(close=lambda:None)), \
                 patch.object(original,'ReactiveController',Controller), \
                 patch.object(original,'evaluate_one',side_effect=lambda c,*a,**kw:dict(success_rate=c.hit,mean_final_distance=1.)), \
                 patch.object(original,'aggregate',side_effect=lambda rows:rows[0]), \
                 patch.object(original,'train_one',side_effect=train), \
                 patch.object(original,'evaluate_method',side_effect=lambda a,r,m,s,split,**kw:[score(kw['fixed_sector'],s)]),redirect_stdout(io.StringIO()):
                original.calibrate_command(args,root)
            info=dict(version=original.VERSION,signature_version=original.CALIBRATION_SIGNATURE_VERSION,
                      signature='sig',seeds=[1,2],b1_grid=cfgs)
            b1=[dict(candidate_index=i,config=c,success_rate=c['b1_hit_threshold'],mean_final_distance=1.) for i,c in enumerate(cfgs)]
            b2={(sector,seed):dict(seed=seed,**score(sector,seed),earliest_checkpoint=102400) for sector in range(10) for seed in (1,2)}
            self.assertEqual(aggregate_b1(info,list(reversed(b1))),read(root/'calibration_b1.json'))
            self.assertEqual(aggregate_b2(info,b2),read(root/'calibration_b2.json'))


class RealParallelWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name);cls.bundle=cls.base/'sealed'
        for folder in ('dual_whisker_rl','scripts'):
            for p in (ROOT/folder).rglob('*.py'):
                dest=cls.bundle/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        config=cls.bundle/'configs/qa.yaml';config.parent.mkdir()
        config.write_text(yaml.safe_dump(dict(plume_model='dynamic',world_bounds=[-2.5,2.5],max_steps=8,
            domain_randomization=False,plume_overrides={'puff_warmup_steps':2,'max_puffs':30})),encoding='utf8')
        command=[sys.executable,str(cls.bundle/'scripts/run_e4_experiments.py'),'prepare','--profile','smoke',
            '--config',str(config),'--output-dir',str(cls.base/'e4'),'--methods','main,b5_history_mlp,b2_fixed',
            '--timesteps','128','--n-envs','1','--max-steps','8','--seeds','1']
        subprocess.run(command,cwd=cls.bundle,check=True,capture_output=True)
        cls.root=cls.base/'e4/smoke';cls.worker=cls.bundle/'scripts/e4_parallel_worker.py'
        cls.payload=dict(root=str(cls.root),config=str(config),target=128,manifest_sha256=sha(cls.root/'manifest.json'),
            worker_sha256=sha(cls.worker),cache_root=str(cls.base/'parallel'))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_real_two_models_and_cache_no_duplicate_training(self):
        q=Queue(self.base/'parallel',self.worker,self.bundle,interval=.05)
        jobs=[make_job(self.payload,'real_'+method,'train',method=method,seed=1,family=family,
              task_dir=str(self.root/method/'seed_1')) for method,family in [('main','G'),('b5_history_mlp','M')]]
        q.run(jobs,limit=2,resource_guard=False)
        for job in jobs:
            self.assertEqual(model_header(Path(job['task_dir'])/'model.zip')['evaluated_checkpoint_steps'],128)
            self.assertEqual(read(Path(job['task_dir'])/'metadata.json')['requested_timesteps'],128)
            from scripts.e4_task_queue import completed_model
            self.assertTrue(completed_model(job))
            self.assertFalse(completed_model({**job,'target':8_000_000}))
            self.assertFalse(completed_model({**job,'expected_signature':'different_source'}))
        with patch.object(q,'start',side_effect=AssertionError('已完成模型不得重训')):
            self.assertEqual(q.run(jobs,limit=2,resource_guard=False)['reused'],2)

    def test_real_rule_candidate_serial_and_parallel_identical(self):
        from scripts.run_e4_experiments import b1_calibration_grid
        q=Queue(self.base/'parallel',self.worker,self.bundle,interval=.05)
        serial=[make_job(self.payload,f'rule_s{i}','b1',candidate_index=i,candidate_config=c,family='G') for i,c in enumerate(b1_calibration_grid()[:2])]
        parallel=[make_job(self.payload,f'rule_p{i}','b1',candidate_index=i,candidate_config=c,family='G') for i,c in enumerate(b1_calibration_grid()[:2])]
        q.run(serial,limit=1,resource_guard=False);q.run(parallel,limit=2,resource_guard=False)
        self.assertEqual([read_result(j) for j in serial],[read_result(j) for j in parallel])

    def test_real_probe_excludes_warmup_and_preserves_formal_model(self):
        q=Queue(self.base/'parallel',self.worker,self.bundle,interval=.05)
        p=self.root/'main/seed_1/model.zip';before=sha(p) if p.exists() else None
        result=probe_case({**self.payload,'target':8_000_000},q,'G',.1)
        self.assertTrue(result['valid'],result)
        self.assertGreater(result['steps'],0);self.assertGreater(result['actors'][0]['first'],0)
        if before:self.assertEqual(sha(p),before)


if __name__=='__main__':unittest.main()
