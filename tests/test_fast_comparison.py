"""首次到源转换、缓存、并行评估与资源选择的实际连通性验证。"""
from __future__ import annotations
import copy
import gzip
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
from dual_whisker_rl.agents import GRUHistoryExtractor
from dual_whisker_rl.arrival_protocol import (
    ROOT, BASELINES, KNOWN_RUNNERS, atomic_json, atomic_trace, cached, compute_metrics,
    convert_job, describe_sources, digest, make_jobs, read, sha,
)
from dual_whisker_rl.arrival_runtime import (
    bounded_map, cpu_episode, cpu_init, gpu_evaluate, make_env, resource_limits, choose_execution, benchmark_cpu,
)
from scripts import run_fast_comparison as runner


def scenario(index, split='test'):
    return dict(scenario_id=f'{split}_{index}', split=split, seed=1000+index+(100 if split=='validation' else 0),
        wind_speed=.16, wind_direction=0., source_position=[0., 0.], robot_pose=[.3+.1*index, 0., math.pi],
        source_strength=1., distance_bin='near', crosswind_bin='center', heading_quadrant=0, repeat=index)


def config(hold=1.):
    return dict(plume_model='dynamic', scenario_mode='randomized', world_bounds=[-2.5, 2.5],
        wind_speed_range=[.12, .2], max_steps=8, dt=.2, success_hold_s=hold,
        domain_randomization=False, plume_overrides={'puff_warmup_steps':2, 'max_puffs':30})


def trajectory(env, action, scene):
    obs, info = env.reset(seed=scene['seed'], options={'scenario':scene})
    initial = info['distance_to_source']; rows=[]
    while True:
        obs, reward, term, trunc, info = env.step(action)
        r=copy.deepcopy(env.unwrapped.trajectory[-1]);r.update(step=env.unwrapped.step_count,elapsed_s=env.unwrapped.step_count*env.unwrapped.dt)
        rows.append(r)
        if term or trunc:break
    return rows, initial


class ArrivalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory()
        cls.root=Path(cls.storage.name)
        cls.source=cls.root/'legacy';cls.source.mkdir()
        cls.primary=cls.root/'primary.zip'
        cls.cfg=config()
        cls.variants={
            'main':{'method':'main','encoder':'gru','history_length':20,'config':{'domain_randomization':False}},
            'fixed':{'method':'b2_fixed','encoder':'gru','history_length':20,'fixed_sector':5,'config':{'domain_randomization':True}},
            'periodic':{'method':'b3_scan','encoder':'gru','history_length':20,'scan_hold':2,'config':{'domain_randomization':True}},
            'single_frame':{'method':'b4_single_frame','encoder':'mlp','history_length':1,'config':{'domain_randomization':True}},
            'history_mlp':{'method':'b5_history_mlp','encoder':'mlp','history_length':20,'config':{'domain_randomization':True}},
            'no_dr':{'method':'main','encoder':'gru','history_length':20,'config':{'domain_randomization':False}}}
        scenes={'test':[scenario(0),scenario(1)],'validation':[scenario(0,'validation')]}
        scientific={p:sha(ROOT/p) for p in ['dual_whisker_rl/envs/mobile_whisker_env.py',
            'dual_whisker_rl/envs/whisker_only_env.py','dual_whisker_rl/experiment_env.py','dual_whisker_rl/e4_experiments.py']}
        unsigned=dict(version=2,scope='simulation_only',profile='formal',base_config=cls.cfg,
            settings={'seeds':[1],'test_scenarios':2,'validation_scenarios':1},variants=cls.variants,
            scenarios=scenes,code_sha256={**scientific,'scripts/run_red_experiments.py':sorted(KNOWN_RUNNERS)[0]})
        cls.manifest={**unsigned,'signature':digest(unsigned)}
        atomic_json(cls.source/'manifest.json',cls.manifest)
        for split,data in scenes.items():atomic_json(cls.source/f'scenarios_{split}.json',data)
        for name in ('main',*BASELINES):
            variant=cls.variants[name]
            env=make_env(cls.cfg,variant,1)
            policy_kwargs={'net_arch':[16,16]}
            if variant['encoder']=='gru':
                policy_kwargs.update(features_extractor_class=GRUHistoryExtractor,
                    features_extractor_kwargs={'history_length':20,'base_observation_dim':16})
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=1,seed=1,
                      policy_kwargs=policy_kwargs,device='cpu')
            if name=='main':
                model.learn(16)  # 只用于软件smoke的16步人工测试模型。
                model.save(cls.primary);env.close();continue
            d=cls.source/name/'seed_1';d.mkdir(parents=True)
            model.save(d/'selected_model.zip');h=sha(d/'selected_model.zip')
            atomic_json(d/'state.json',dict(status='completed',signature=cls.manifest['signature'],
                actual_timesteps=0,selected_model_sha256=h))
            for scene in scenes['test']:
                action=0 if name in ('fixed','periodic') else np.array([0,5,5])
                rows,initial=trajectory(env,action,scene)
                metrics=compute_metrics(rows,initial,scene,{**cls.cfg,'goal_radius':.1})
                metrics.update(success=False,termination='timeout',steps=len(rows),variant=name,training_seed=1,
                    fault='nominal',model_sha256=h,evaluation_scene_count=2,
                    evaluation_signature=digest(dict(manifest=cls.manifest['signature'],model=h,fault={},scenes=scenes['test'])))
                p=d/'episodes/nominal'/f"{scene['scenario_id']}.json"
                atomic_json(p,metrics);atomic_trace(p.with_name(p.stem+'.steps.jsonl.gz'),rows)
            env.close()

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()

    def spec_jobs(self,out):
        spec=describe_sources(self.source,self.primary,out)
        return spec,make_jobs(spec,out)

    def test_boundary_and_transient_arrival_conversion_matches_direct(self):
        scene=scenario(0);scene['robot_pose']=[.115,0.,math.pi]
        env=make_env(config(1),self.variants['main'],1)
        obs,info=env.reset(seed=scene['seed'],options={'scenario':scene});initial=info['distance_to_source']
        rows=[]
        for _ in range(8):
            _,_,_,_,_=env.step([0,5,5])
            r=copy.deepcopy(env.unwrapped.trajectory[-1]);r.update(step=env.unwrapped.step_count,elapsed_s=env.unwrapped.step_count*.2);rows.append(r)
        env.close()
        self.assertLess(rows[0]['distance_to_source'],.1)
        self.assertGreater(rows[-1]['distance_to_source'],.1)
        with tempfile.TemporaryDirectory() as tmp:
            spec,jobs=self.spec_jobs(Path(tmp)/'out');job=next(j for j in jobs if j['model']['variant']=='no_dr')
            job=copy.deepcopy(job);job['scene']=scene
            p=Path(tmp)/'old.json';g=Path(tmp)/'old.gz'
            old=dict(model_sha256=job['model']['sha256'],scenario_id=scene['scenario_id'],scenario_seed=scene['seed'],
                training_seed=1,variant='no_dr',fault='nominal',evaluation_scene_count=2,
                evaluation_signature=job['legacy_evaluation_signature'],initial_distance=initial,steps=len(rows),termination='timeout')
            atomic_json(p,old);atomic_trace(g,rows)
            job['source']=dict(row=str(p),trace=str(g),row_sha256=sha(p),trace_sha256=sha(g))
            result=convert_job(job);self.assertEqual(result['status'],'converted')
            converted=read(job['output'])
            direct=make_env(config(0),self.variants['main'],1)
            direct_rows,distance=trajectory(direct,[0,5,5],scene);direct.close()
            expected=compute_metrics(direct_rows,distance,scene,job['config'])
            for k,value in expected.items():
                if isinstance(value,float):self.assertAlmostEqual(converted[k],value,places=10,msg=k)
                else:self.assertEqual(converted[k],value,k)
            self.assertEqual(converted['steps'],1)
            boundary=copy.deepcopy(direct_rows);boundary[-1]['distance_to_source']=.1
            self.assertTrue(compute_metrics(boundary,distance,scene,job['config'])['success'])
            boundary[-1]['distance_to_source']=.100000001
            self.assertFalse(compute_metrics(boundary,distance,scene,job['config'])['success'])

    def test_all_historical_convert_without_torch_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,jobs=self.spec_jobs(Path(tmp)/'out')
            old_hash=sha(self.source/'manifest.json')
            results=list(bounded_map(convert_job,[j for j in jobs if j['source']],2))
            self.assertTrue(all(r['status']=='converted' for r in results))
            self.assertEqual(sha(self.source/'manifest.json'),old_hash)
            self.assertTrue(all(cached(j) for j in jobs if j['source']))

    def test_cpu_parallel_and_central_batch_results_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,jobs=self.spec_jobs(Path(tmp)/'one');main=[j for j in jobs if j['model']['variant']=='main'];m=spec['models'][0]
            for path in bounded_map(cpu_episode,main,1,initializer=cpu_init,initargs=(m,spec['config'],m['variant_config'])):pass
            first=[read(j['output']) for j in main]
            second=copy.deepcopy(main)
            for j in second:j['output']=str(Path(tmp)/'two'/Path(j['output']).name)
            for path in bounded_map(cpu_episode,second,2,initializer=cpu_init,initargs=(m,spec['config'],m['variant_config'])):pass
            third=copy.deepcopy(main)
            for j in third:j['output']=str(Path(tmp)/'batch'/Path(j['output']).name)
            # 无CUDA机器也验证集中批量协调器的slot、动作、终止和场景调度。
            gpu_evaluate(third,m,spec['config'],m['variant_config'],2,'cpu')
            for i in range(len(main)):
                for k in ('success','steps','final_distance','odor_hit_ratio','reward_components','scenario_id'):
                    self.assertEqual(first[i][k],read(second[i]['output'])[k],k)
                    self.assertEqual(first[i][k],read(third[i]['output'])[k],k)

    def test_cache_corruption_and_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,jobs=self.spec_jobs(Path(tmp)/'out');job=next(j for j in jobs if j['source'])
            convert_job(job);self.assertTrue(cached(job))
            row=read(job['output']);row['success']=not row['success'];atomic_json(job['output'],row)
            self.assertFalse(cached(job))
            convert_job(job);altered=copy.deepcopy(job);altered['signature']='different'
            with self.assertRaises(ValueError):cached(altered)
            altered=copy.deepcopy(job);altered['source']['trace_sha256']='bad'
            self.assertEqual(convert_job(altered)['status'],'needs_evaluation')

    def test_plan_no_writes_and_changed_bank_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            out=Path(tmp)/'out'
            args=runner.parser().parse_args(['plan','--reuse-from',str(self.source),'--model-path',str(self.primary),'--output-dir',str(out)])
            spec=runner.resolve(args);preview=runner.preview(spec,out)
            self.assertEqual(preview['training_tasks'],0);self.assertEqual(preview['expected_total_episodes'],12)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):describe_sources(self.source,self.primary,self.source/'child')

    def test_full_run_resume_skips_model_and_missing_scene_only(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            out=Path(tmp)/'out'
            args=runner.parser().parse_args(['run','--reuse-from',str(self.source),'--model-path',str(self.primary),
                '--output-dir',str(out),'--workers','1','--device','cpu','--calibration-seconds','0','--bootstrap','20'])
            spec=runner.resolve(args)
            with patch.object(PPO,'learn',side_effect=AssertionError('正式入口不应训练')):
                result=runner.execute(args,spec)
            self.assertEqual(result['completed'],12)
            with patch('dual_whisker_rl.arrival_runtime.load_policy',side_effect=AssertionError('全缓存不应加载模型')):
                runner.execute(args,spec)
            jobs=make_jobs(spec,out);main=[j for j in jobs if j['model']['variant']=='main']
            Path(main[0]['output']).unlink()
            # 只删除一个场景；恢复后其余场景摘要不变。
            keep=sha(main[1]['output']);runner.execute(args,spec)
            self.assertEqual(sha(main[1]['output']),keep)
            self.assertEqual(read(out/'state.json')['completed'],12)
            self.assertEqual(len(list((out/'main').glob('seed_*/*.json'))),2)

    def test_timeout_oob_and_arrival_on_last_step(self):
        scene=scenario(0);scene['robot_pose']=[.335,0.,math.pi]
        env=make_env(config(0),self.variants['main'],1)
        rows,initial=trajectory(env,[0,5,5],scene);env.close()
        value=compute_metrics(rows,initial,scene,{**config(0),'goal_radius':.1})
        self.assertTrue(value['success']);self.assertEqual(value['steps'],8)
        self.assertAlmostEqual(value['elapsed_s'],1.6)
        scene['robot_pose']=[1.,0.,0.]
        env=make_env(config(0),self.variants['main'],1)
        rows,initial=trajectory(env,[5,5,5],scene);env.close()
        value=compute_metrics(rows,initial,scene,{**config(0),'goal_radius':.1})
        self.assertEqual(value['termination'],'timeout');self.assertEqual(value['penalized_time_s'],1.6)
        scene['robot_pose']=[2.49,0.,0.]
        env=make_env(config(0),self.variants['main'],1)
        rows,initial=trajectory(env,[0,5,5],scene);env.close()
        value=compute_metrics(rows,initial,scene,{**config(0),'goal_radius':.1})
        self.assertEqual(value['termination'],'out_of_bounds');self.assertEqual(value['steps'],1)
        self.assertEqual(value['penalized_time_s'],1.6)

    def test_corrupt_trace_repairs_only_that_scene_and_no_cuda_does_not_start(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            source=Path(tmp)/'source';shutil.copytree(self.source,source)
            broken=next((source/'fixed/seed_1/episodes/nominal').glob('*.gz'))
            broken.write_bytes(b'broken gzip')
            out=Path(tmp)/'out'
            args=runner.parser().parse_args(['run','--reuse-from',str(source),'--model-path',str(self.primary),
                '--output-dir',str(out),'--workers','1','--device','auto','--calibration-seconds','0','--bootstrap','10'])
            spec=runner.resolve(args)
            with patch('dual_whisker_rl.arrival_runtime.resource_limits',return_value={'cuda_devices':[]}):
                with self.assertRaisesRegex(RuntimeError,'无卡'):runner.execute(args,spec)
            self.assertFalse(out.exists())
            args.device='cpu'
            runner.execute(args,spec)
            repairs=read(out/'repairs.json');self.assertEqual(len(repairs),1)
            result=read(repairs[0]['output']);self.assertEqual(result['result_origin'],'evaluated_cpu')
            self.assertEqual(len(list((out/'fixed/seed_1').glob('*.json'))),2)
            altered=read(source/'scenarios_test.json');altered[0]['seed']+=999
            atomic_json(source/'scenarios_test.json',altered)
            with self.assertRaisesRegex(ValueError,'题库'):runner.resolve(args)

    def test_cpu_throughput_probe_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,jobs=self.spec_jobs(Path(tmp)/'out');m=spec['models'][0]
            result=benchmark_cpu(m,spec['config'],m['variant_config'],spec['scenarios']['validation'],1,.05)
            self.assertGreater(result['steps_per_second'],0)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_resource_selection_respects_caps_and_cuda_mismatch(self):
        resources=dict(cpu_workers_cap=4,env_workers_cap=3,cuda_devices=[dict(device='cuda:0',free_bytes=2**30)])
        model=dict(variant_config={},path='unused')
        with patch('dual_whisker_rl.arrival_runtime.cuda_consistency',return_value={'equal':False}):
            choice=choose_execution(resources,model,{},[{}],'1','auto',0)
            self.assertEqual(choice['selected']['device'],'cpu')
            with self.assertRaises(ValueError):choose_execution(resources,model,{},[{}],'1','cuda',0)
        with self.assertRaises(ValueError):choose_execution(resources,model,{},[{}],'8','cpu',0)
        detected=resource_limits(probe_cuda=False)
        self.assertLessEqual(detected['cpu_workers_cap'],detected['effective_cpus'])
        self.assertGreaterEqual(detected['cpu_workers_cap'],1)


if __name__=='__main__':unittest.main()
