"""E4新预算的参数传递、封存隔离与完成判定，不运行正式训练。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dual_whisker_rl.arrival_protocol import atomic_json,read,sha
from scripts.run_remaining_formal import parser
from scripts.run_e4_experiments import ppo_rollout_settings
from scripts.stochastic_campaign import (e4_training_command,validate_e4_training_manifest,
    e4_training_completed,episode_store,preview,synchronize_original)


class E4BudgetOverrideTests(unittest.TestCase):
    def test_default_budget_per_seed_and_existing_mlp_unchanged(self):
        args=parser().parse_args([])
        self.assertEqual(args.e4_timesteps,8_000_000)
        self.assertEqual(args.mlp_timesteps,15_000_000)
        self.assertEqual(parser().parse_args(['--e4-timesteps','4000000']).e4_timesteps,4_000_000)

    def test_budget_explicit_for_calibration_and_training(self):
        args=SimpleNamespace(e4_timesteps=8_000_000,e4_training_dir=Path('/new/e4_8m/formal'))
        for phase in ('calibrate','train'):
            cmd=e4_training_command(args,Path('/sealed'),phase)
            self.assertEqual(cmd[cmd.index('--timesteps')+1],'8000000')
            self.assertEqual(cmd[cmd.index('--output-dir')+1],args.e4_training_dir.parent)
            self.assertIn('--resume',cmd)
        with self.assertRaises(ValueError):e4_training_command(args,Path('/sealed'),'run')

    def test_missing_manifest_is_not_written_by_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(e4_training_dir=Path(tmp)/'formal',e4_timesteps=8_000_000)
            self.assertIsNone(validate_e4_training_manifest(args,{}))
            self.assertFalse(args.e4_training_dir.exists())
            with self.assertRaises(FileNotFoundError):validate_e4_training_manifest(args,{},required=True)

    def test_new_budget_preserves_reference_conditions_and_rejects_old_budget(self):
        reference=dict(timesteps=1_000_000,profile='formal',config={'world_bounds':[-2.5,2.5]},
            method_seeds={'main':[1,2,3,4,5]},scenario_hashes={'test':'frozen'},code_hashes={'science':'original'})
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(e4_training_dir=Path(tmp)/'formal',e4_timesteps=8_000_000)
            p=args.e4_training_dir/'manifest.json'
            atomic_json(p,reference)
            before=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'预算'):validate_e4_training_manifest(args,reference)
            self.assertEqual(before,p.read_bytes())
            atomic_json(p,{**reference,'timesteps':8_000_000})
            self.assertEqual(validate_e4_training_manifest(args,reference)['timesteps'],8_000_000)
            atomic_json(p,{**reference,'timesteps':8_000_000,'config':{'world_bounds':[-1,1]}})
            with self.assertRaisesRegex(ValueError,'config'):validate_e4_training_manifest(args,reference)

    def test_one_million_model_cannot_satisfy_eight_million(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=Path(tmp);(task/'model.zip').touch()
            atomic_json(task/'state.json',dict(status='completed',actual_timesteps=1_003_520))
            atomic_json(task/'metadata.json',dict(requested_timesteps=1_000_000,actual_timesteps=1_003_520))
            with patch('scripts.stochastic_campaign.model_header',side_effect=AssertionError('不应读取旧模型')):
                self.assertFalse(e4_training_completed(task,8_000_000))

    def test_completed_budget_checks_actual_checkpoint_not_status_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=Path(tmp);(task/'model.zip').touch()
            atomic_json(task/'state.json',dict(status='completed',actual_timesteps=8_003_584))
            atomic_json(task/'metadata.json',dict(requested_timesteps=8_000_000,actual_timesteps=8_003_584))
            with patch('scripts.stochastic_campaign.model_header',return_value={'evaluated_checkpoint_steps':8_003_584}):
                self.assertTrue(e4_training_completed(task,8_000_000))
            with patch('scripts.stochastic_campaign.model_header',return_value={'evaluated_checkpoint_steps':1_003_520}):
                self.assertFalse(e4_training_completed(task,8_000_000))

    def test_shared_store_and_rollout_rounding(self):
        args=SimpleNamespace(output_dir=Path('/new/output'),episode_store=Path('/old/episodes'))
        self.assertEqual(episode_store(args),Path('/old/episodes'))
        args.episode_store=None;self.assertEqual(episode_store(args),Path('/new/output/episodes'))
        n,rollout,batch=ppo_rollout_settings(SimpleNamespace(timesteps=8_000_000,n_envs=8))
        self.assertEqual((n,rollout,batch),(512,4096,256))
        self.assertEqual(((8_000_000+rollout-1)//rollout)*rollout,8_003_584)

    def test_plan_counts_seven_methods_five_seeds_and_excludes_rules(self):
        methods=('main','b2_fixed','b3_scan','b4_single_frame','b5_history_mlp','b6_left','b6_right','b0_random')
        e4=dict(method_seeds={n:[1,2,3,4,5] for n in methods},scenario_counts={'test':648,'validation':216})
        e4['method_seeds']['b1_reactive']=[1]
        groups={n:['main'] for n in ('core','history','features','rewards','geometry','sensor','randomization')}
        ctx=dict(old={'settings':{'seeds':[1,2,3,4,5]},'scenarios':{'test':[{}]*648}},
                 spec={'groups':groups},native=['main'],completed=[],e4=e4,sampling={},
                 training={'target_total_steps':15_000_000,'status':'running'})
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(e4_training_dir=Path(tmp)/'formal',e4_timesteps=8_000_000,eval_repeats=1,stages=[],output_dir=Path(tmp)/'new')
            result=preview(args,ctx)
        self.assertEqual(result['remaining_e4_training'],35)
        self.assertEqual(result['e4_target_steps_each'],8_000_000)
        self.assertEqual(result['e4_total_target_steps'],280_000_000)
        self.assertTrue(result['e4_calibration_budget_unchanged'])

    def test_verified_bundle_transport_requires_exact_commit_and_preserves_ff_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bundle=out/'commit.bundle';bundle.write_bytes(b'software test fixture')
            receipt=dict(original_checkout=str(out/'checkout'),release_commit='expected',original_uncommitted_sha256={},
                verified_git_bundle=dict(path=str(bundle),sha256=sha(bundle),commit='expected'))
            atomic_json(out/'deployment.json',receipt);calls=[]
            def git(argv,**kwargs):
                calls.append(argv)
                return 'expected\n' if argv[1]=='rev-parse' else ''
            with patch('subprocess.check_output',side_effect=git):synchronize_original(out)
            self.assertIn(['git','bundle','verify',str(bundle.resolve())],calls)
            self.assertIn(['git','pull','--ff-only',str(bundle.resolve()),'main'],calls)
            self.assertNotIn(['git','fetch','origin','main'],calls)
            self.assertEqual(read(out/'deployment.json')['synchronization_transport'],'verified_git_bundle')

    def test_tampered_bundle_rejected_before_git_or_stash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bundle=out/'commit.bundle';bundle.write_bytes(b'wrong content')
            atomic_json(out/'deployment.json',dict(original_checkout=str(out/'checkout'),release_commit='expected',
                verified_git_bundle=dict(path=str(bundle),sha256='not_actual',commit='expected')))
            with patch('subprocess.check_output',side_effect=AssertionError('不得执行Git')),self.assertRaises(ValueError):
                synchronize_original(out)


if __name__=='__main__':unittest.main()
