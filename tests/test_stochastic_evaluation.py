"""长暖场物理等价、随机采样复现和新旧协议隔离的实际回归。"""
from __future__ import annotations
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from stable_baselines3 import PPO
from dual_whisker_rl.agents import GRUHistoryExtractor
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.experiment_env import ExperimentMobileEnv
from dual_whisker_rl.arrival_protocol import atomic_json,cached,digest,read,sha
from dual_whisker_rl.arrival_runtime import make_env,cpu_init,cpu_episode,bounded_map,gpu_evaluate
from dual_whisker_rl.stochastic_sampling import EpisodeSampler,from_uniforms,probabilities,sampled_actions,policy_seed
from dual_whisker_rl.stochastic_protocol import (ROOT,settings,make_jobs,comparison_spec,effective_environment,
    scientific_hashes,source_manifest,seal,view_rows)


def scene(i,split='test'):
    return dict(scenario_id=f'{split}_{i}',split=split,seed=800+i+(200 if split=='validation' else 0),
        wind_speed=.16,wind_direction=0.,source_position=[-1.,0.],robot_pose=[.4+i*.05,0.,math.pi],
        source_strength=1.,distance_bin='near',crosswind_bin='center',heading_quadrant=0,repeat=i)


def cfg():
    return dict(plume_model='dynamic',world_bounds=[-2.5,2.5],wind_speed_range=[.12,.2],
        scenario_mode='randomized',max_steps=8,dt=.2,success_hold_s=0.,goal_radius=.1,
        domain_randomization=False,include_blank_age_observation=True,
        plume_overrides={'puff_warmup_steps':2,'max_puffs':30})


def trace(job):
    with gzip.open(Path(job['output']).with_name(Path(job['output']).stem+'.steps.jsonl.gz'),'rt') as f:
        return [json.loads(line) for line in f]


class PhysicsPropagationTests(unittest.TestCase):
    def test_long_warmup_default_and_explicit_parameters_equivalent(self):
        for overrides,half,wind,dr in [({},2.5,.12,False),({},4.,.08,False),
                ({'max_puff_age':12.,'puff_bounds_margin':2.4},2.5,.12,False),({},2.5,.12,True)]:
            config=cfg();config.update(world_bounds=[-half,half],wind_speed_range=[wind,wind+.04],
                max_steps=90,domain_randomization=dr)
            config['plume_overrides'].update(puff_warmup_steps=180,max_puffs=500,**overrides)
            base=MobileWhiskerPuffEnv(config);experiment=ExperimentMobileEnv(config)
            self.assertEqual(base.plume.max_puff_age,experiment.plume.max_puff_age)
            self.assertEqual(base.plume._dr_nominal,experiment.plume._dr_nominal)
            self.assertEqual(base.plume.puff_bounds_margin,experiment.plume.puff_bounds_margin)
            for seed in (11,72):
                a,_=base.reset(seed=seed);b,_=experiment.reset(seed=seed)
                np.testing.assert_array_equal(a,b)
                for i in range(75):
                    action=[5,i%10,9-i%10]
                    left,right=base.step(action),experiment.step(action)
                    np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:4],right[1:4])
                    self.assertEqual(base.plume.puffs,experiment.plume.puffs)
            base.close();experiment.close()

    def test_factory_does_not_hardcode_lifetime_and_keeps_defaults(self):
        config=cfg();a=MobileWhiskerPuffEnv(config);b=ExperimentMobileEnv(config)
        self.assertAlmostEqual(b.plume.max_puff_age,45.833333333333336)
        self.assertEqual(type(a.plume).__name__,'DynamicPuffPlume')
        self.assertEqual(type(b.plume).__name__,'ExperimentPuff')
        self.assertNotIn('max_puff_age',config['plume_overrides'])
        a.close();b.close()


class StochasticEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.source=cls.root/'legacy';cls.sealed=cls.root/'sealed';cls.source.mkdir();cls.sealed.mkdir()
        cls.primary=cls.root/'models/main.zip';cls.primary.parent.mkdir()
        cls.sampling=settings();cls.config=cfg()
        cls.variants={
            'main':dict(method='main',encoder='gru',history_length=20,config={'domain_randomization':False}),
            'fixed':dict(method='b2_fixed',encoder='gru',history_length=20,fixed_sector=5,config={'domain_randomization':True}),
            'periodic':dict(method='b3_scan',encoder='gru',history_length=20,scan_hold=2,config={'domain_randomization':True}),
            'single_frame':dict(method='b4_single_frame',encoder='mlp',history_length=1,config={'domain_randomization':True}),
            'history_mlp':dict(method='b5_history_mlp',encoder='mlp',history_length=20,config={'domain_randomization':True}),
            'no_dr':dict(method='main',encoder='gru',history_length=20,config={'domain_randomization':False})}
        names=['dual_whisker_rl/envs/whisker_only_env.py','dual_whisker_rl/experiment_env.py','scripts/run_red_experiments.py']
        for n in names:
            p=cls.sealed/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        manifest=dict(version=2,scope='simulation_only',base_config=cls.config,
            settings={'seeds':[1],'timesteps':0,'test_scenarios':3,'validation_scenarios':2},
            variants=cls.variants,scenarios={'test':[scene(i) for i in range(3)],'validation':[scene(i,'validation') for i in range(2)]},
            code_sha256={n:sha(cls.sealed/n) for n in names})
        cls.manifest=dict(manifest,signature=digest(manifest));atomic_json(cls.source/'manifest.json',cls.manifest)
        for split,bank in manifest['scenarios'].items():atomic_json(cls.source/f'scenarios_{split}.json',bank)
        for name,var in cls.variants.items():
            env=make_env(cls.config,var,1);kw={'net_arch':[16,16]}
            if var['encoder']=='gru':kw.update(features_extractor_class=GRUHistoryExtractor,
                features_extractor_kwargs={'history_length':20,'base_observation_dim':16})
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=1,seed=1,policy_kwargs=kw,device='cpu')
            task=cls.source/name/'seed_1';task.mkdir(parents=True)
            model.save(task/'selected_model.zip');shutil.copyfile(task/'selected_model.zip',task/'final_model.zip')
            atomic_json(task/'state.json',dict(status='completed',signature=cls.manifest['signature'],
                actual_timesteps=0,selected_model_sha256=sha(task/'selected_model.zip')))
            if name=='main':shutil.copyfile(task/'selected_model.zip',cls.primary)
            env.close()
        atomic_json(cls.root/'logs/main/run_metadata.json',dict(final_num_timesteps=0,base_observation_dim=16,
            history_length=20,temporal_encoder='gru',config=cls.config))
        cls.spec=comparison_spec(cls.source,cls.primary,cls.sealed,cls.sampling)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def jobs(self,label='one',spec=None):
        return make_jobs(spec or {**self.spec,'models':self.spec['models'][:1]},self.root/label)

    def test_inverse_cdf_boundaries_and_multiple_heads(self):
        heads=[np.array([[.2,.8],[.2,.8],[.2,.8]]),np.ones((3,5))/5]
        a=from_uniforms(heads,[[0.,0.],[.2,.2],[.999,.999]])
        np.testing.assert_array_equal(a,[[0,0],[1,1],[1,4]])
        with self.assertRaises(ValueError):from_uniforms(heads,[[1.,0.]]*3)

    def test_seeds_independent_of_order_and_fault_but_repeat_changes(self):
        seeds=[policy_seed(self.sampling,'sha',scene(i)) for i in range(3)]
        self.assertEqual(seeds,list(reversed([policy_seed(self.sampling,'sha',scene(i)) for i in reversed(range(3))])))
        self.assertNotEqual(seeds[0],policy_seed(self.sampling,'sha',scene(0),1))
        a=EpisodeSampler(self.sampling,'sha',scene(0));b=EpisodeSampler(self.sampling,'sha',scene(0))
        np.testing.assert_array_equal(a.rng.random(40),b.rng.random(40))

    def test_discrete_and_multidiscrete_contracts(self):
        for model in self.spec['models']:
            config,physics=effective_environment(self.config,model['variant_config'])
            self.assertTrue(physics['action_nvec']);self.assertFalse(config['domain_randomization'])

    def test_single_multi_and_batched_environment_results_match(self):
        model=self.spec['models'][0];one=self.jobs('serial');many=self.jobs('parallel');batch=self.jobs('batch')
        cpu_init(model,self.config,model['variant_config'])
        for job in one:cpu_episode(job)
        list(bounded_map(cpu_episode,list(reversed(many)),2,initializer=cpu_init,
                         initargs=(model,self.config,model['variant_config'])))
        gpu_evaluate(list(reversed(batch)),model,self.config,model['variant_config'],2,'cpu')
        for a,b,c in zip(one,many,batch):
            self.assertEqual(trace(a),trace(b));self.assertEqual(trace(a),trace(c))
            self.assertTrue(cached(a));self.assertEqual(read(a['output'])['protocol'],'stochastic-arrival-v2')

    def test_cache_full_no_model_load_and_partial_only_missing(self):
        from dual_whisker_rl.stochastic_runtime import run_jobs
        import argparse
        jobs=self.jobs('resume');model=self.spec['models'][0]
        args=argparse.Namespace(workers='1',device='cpu',calibration_seconds=0.,bootstrap=5)
        run_jobs(args,jobs,self.spec['scenarios']['validation'],self.root/'resume_run')
        with patch('dual_whisker_rl.arrival_runtime.load_policy',side_effect=AssertionError('不应加载')):
            result=run_jobs(args,jobs+jobs,self.spec['scenarios']['validation'],self.root/'resume_run')
        self.assertEqual(result['evaluated'],0);self.assertEqual(result['unique'],3)
        Path(jobs[1]['output']).unlink()
        result=run_jobs(args,jobs,self.spec['scenarios']['validation'],self.root/'resume_run')
        self.assertEqual(result['evaluated'],1)

    def test_signature_rejects_old_protocol_changed_model_bank_or_sampling(self):
        jobs=self.jobs('change');old=copy.deepcopy(jobs[0])
        atomic_json(old['output'],dict(job_signature='legacy-v1'))
        with self.assertRaises(ValueError):cached(old)
        for key in ('model','bank','sampling'):
            changed=copy.deepcopy(self.spec)
            if key=='model':changed['models'][0]['sha256']='changed'
            elif key=='bank':changed['scenarios']['test'][0]['seed']+=1
            else:changed['sampling']['eval_seed']+=1
            self.assertNotEqual(jobs[0]['signature'],self.jobs('change',changed)[0]['signature'])

    def test_shared_reports_same_jobs_fault_streams_and_repeat_identity(self):
        a=self.jobs();b=self.jobs(spec={**self.spec,'models':self.spec['models'][:1]})
        self.assertEqual([j['signature'] for j in a],[j['signature'] for j in b])
        fault=make_jobs({**self.spec,'models':self.spec['models'][:1]},self.root/'one',fault={'action_delay_steps':2},label='delay')
        self.assertNotEqual(a[0]['signature'],fault[0]['signature'])
        self.assertEqual(policy_seed(a[0]['sampling'],a[0]['model']['sha256'],a[0]['scene']),
                         policy_seed(fault[0]['sampling'],fault[0]['model']['sha256'],fault[0]['scene']))
        doubled=copy.deepcopy(self.spec);doubled['sampling']=settings(repeats=2)
        jobs=make_jobs(doubled,self.root/'repeats');self.assertEqual(len(jobs),36)
        self.assertEqual(len({j['signature'] for j in jobs[:6]}),6)
        self.assertLessEqual(len({j['signature'] for j in jobs}),len(jobs))

    def test_sealed_source_tampering_refused_without_modifying_original(self):
        bad=self.root/'bad_seal';shutil.copytree(self.sealed,bad,dirs_exist_ok=True)
        (bad/'dual_whisker_rl/experiment_env.py').write_text('# altered')
        with self.assertRaises(FileNotFoundError):source_manifest(self.source,bad)
        source_manifest(self.source,self.sealed)

    def test_sampler_uses_sb3_probabilities_with_no_global_torch_rng(self):
        import torch
        model=PPO.load(self.primary,device='cpu');env=make_env(self.config,self.variants['main'],1)
        obs,_=env.reset(seed=1)
        a=EpisodeSampler(self.sampling,sha(self.primary),scene(0));b=EpisodeSampler(self.sampling,sha(self.primary),scene(0))
        torch.manual_seed(1);left=sampled_actions(model,np.asarray([obs]),[a])
        torch.manual_seed(9797);right=sampled_actions(model,np.asarray([obs]),[b])
        np.testing.assert_array_equal(left,right)
        p=probabilities(model,np.asarray([obs]));self.assertEqual([x.shape[1] for x in p],[6,10,10])
        env.close()

    def test_all_cached_main_report_is_complete_and_new_protocol(self):
        from dual_whisker_rl.arrival_reports import make_report
        spec={**self.spec,'models':self.spec['models'][:1]};jobs=self.jobs('report',spec)
        model=spec['models'][0];cpu_init(model,self.config,model['variant_config'])
        for job in jobs:cpu_episode(job)
        report=make_report(spec,jobs,self.root/'report_view',draws=5)
        manifest=read(report.parent/'report_manifest.json')
        self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['protocol'],'stochastic-arrival-v2')
        self.assertIn('随机采样',report.read_text(encoding='utf8'))

    def test_new_cli_plan_is_read_only_and_does_not_load_policy(self):
        from scripts.run_fast_comparison import parser
        from scripts.stochastic_campaign import fast
        out=self.root/'readonly_plan'
        args=parser().parse_args(['plan','--reuse-from',str(self.source),'--model-path',str(self.primary),
            '--sealed-source',str(self.sealed),'--output-dir',str(out)])
        with patch('dual_whisker_rl.arrival_runtime.load_policy',side_effect=AssertionError('不应加载')):
            result=fast(args)
        self.assertEqual(result['expected'],18);self.assertFalse(out.exists())

    def test_adoption_of_completed_training_never_calls_training(self):
        from scripts.stochastic_campaign import adopt_training
        import argparse
        training=self.root/'adoption_training';out=self.root/'adoption_output'
        record=dict(status='completed',model_sha256='frozen',training_config={},model_path='untouched')
        atomic_json(training/'mlp_training.json',record)
        args=argparse.Namespace(adopt_training_run=training,output_dir=out,mlp_timesteps=15_000_000)
        with patch('scripts.run_remaining_formal.validate_mlp',return_value={'sha256':'frozen'}), \
             patch('scripts.train_mobile_whisker_ppo.train',side_effect=AssertionError('不得训练')):
            adopt_training(args,{})
        self.assertEqual(read(out/'adopted_training.json')['record'],record)
        self.assertEqual(read(training/'mlp_training.json'),record)

    def test_incomplete_adopted_training_stops_without_resuming(self):
        from scripts.stochastic_campaign import adopt_training
        import argparse
        training=self.root/'incomplete_training';out=self.root/'incomplete_output'
        atomic_json(training/'mlp_training.json',dict(status='running'))
        args=argparse.Namespace(adopt_training_run=training,output_dir=out,mlp_timesteps=15_000_000)
        with patch('scripts.train_mobile_whisker_ppo.train',side_effect=AssertionError('不得训练')),self.assertRaises(RuntimeError):
            adopt_training(args,{})

    def test_all_six_methods_end_to_end_and_repeated_run_reuses(self):
        from scripts.run_fast_comparison import parser
        from scripts.stochastic_campaign import fast
        args=parser().parse_args(['run','--reuse-from',str(self.source),'--model-path',str(self.primary),
            '--sealed-source',str(self.sealed),'--output-dir',str(self.root/'fast_run'),'--workers','1',
            '--device','cpu','--calibration-seconds','0','--bootstrap','5'])
        result=fast(args);self.assertEqual(result['status'],'completed')
        with patch('dual_whisker_rl.arrival_runtime.load_policy',side_effect=AssertionError('不得加载')):
            result=fast(args)
        self.assertEqual(result['evaluated'],0)

    def test_campaign_reads_existing_chinese_wrapper_as_utf8(self):
        import argparse
        from scripts.stochastic_campaign import run
        from scripts.run_remaining_formal import FROZEN_WORKER
        out=self.root/'utf8_campaign';bundle=out/'sealed_v2'
        worker=bundle/'scripts/remaining_worker.py';worker.parent.mkdir(parents=True)
        worker.write_text(FROZEN_WORKER,encoding='utf8')
        args=argparse.Namespace(phase='run',output_dir=out,stages=[])
        ctx=dict(signed={},files={},old={},spec={'groups':{}})
        with patch('scripts.stochastic_campaign.context',return_value=ctx), \
             patch('scripts.stochastic_campaign.preview',return_value={}), \
             patch('scripts.remaining_formal_support.create_bundle'):
            result=run(args)
        self.assertEqual(result['status'],'selected_stages_completed')


if __name__=='__main__':unittest.main()
