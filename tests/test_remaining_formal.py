"""验证预算恢复、封存隔离、进程串联与逐局完整性，避免正式任务重复计算。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from dual_whisker_rl.arrival_protocol import atomic_json, digest, read
from scripts import run_remaining_formal as runner
from scripts.remaining_formal_support import (
    campaign_lock, create_bundle, monitored_run, reject_live_children, verified_manifest,
)


class RemainingFormalTests(unittest.TestCase):
    def test_plan_defaults_keep_long_model_and_formal_protocols_separate(self):
        args = runner.resolve(runner.parser().parse_args([]))
        self.assertEqual(args.phase, 'plan')
        self.assertEqual(args.mlp_timesteps, 15_000_000)
        self.assertEqual(args.mlp_n_envs, 12)
        self.assertEqual(args.stages, list(runner.STAGES))
        for flags in (['--mlp-timesteps', '0'], ['--stages', 'e3,e3'],
                      ['--output-dir', 'results/simulation_experiments/formal/child']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                runner.resolve(runner.parser().parse_args(flags))

    def test_manifest_rejects_source_or_scenario_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unsigned = dict(version=2, scope='simulation_only', scenarios={'test': [1], 'validation': [2]})
            atomic_json(root/'manifest.json', dict(unsigned, signature=digest(unsigned)))
            for split, rows in unsigned['scenarios'].items():
                atomic_json(root/f'scenarios_{split}.json', rows)
            verified_manifest(root)
            atomic_json(root/'scenarios_test.json', [3])
            with self.assertRaisesRegex(ValueError, '场景'):
                verified_manifest(root)
            atomic_json(root/'manifest.json', dict(unsigned, signature='changed'))
            with self.assertRaisesRegex(ValueError, '签名'):
                verified_manifest(root)

    def test_sealed_bundle_preserves_original_file_set_and_rejects_extra_module(self):
        from dual_whisker_rl.arrival_protocol import sha
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base/'source'
            (source/'dual_whisker_rl').mkdir(parents=True)
            (source/'scripts').mkdir()
            (source/'dual_whisker_rl/__init__.py').write_text('# 科学代码\n', encoding='utf-8')
            (source/'scripts/run_red_experiments.py').write_text('# 封存入口\n', encoding='utf-8')
            names = ['dual_whisker_rl/__init__.py', 'scripts/run_red_experiments.py']
            manifest = dict(code_sha256={n:sha(source/n) for n in names}, base_config={}, settings={}, variants={}, faults={})
            target = base/'bundle'
            create_bundle(target, {n:source/n for n in names}, manifest, {})
            (target/'dual_whisker_rl/extra.py').write_text('')
            with self.assertRaisesRegex(ValueError, '额外'):
                create_bundle(target, {n:source/n for n in names}, manifest, {})

    def test_monitor_records_success_and_crash_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = monitored_run([sys.executable, '-c', "print('progress',flush=True)"],
                root, root/'success.log', root/'success.json', interval=.01, stall_seconds=10)
            self.assertEqual(receipt['status'], 'completed')
            with self.assertRaises(__import__('subprocess').CalledProcessError):
                monitored_run([sys.executable, '-c', 'raise SystemExit(7)'], root,
                    root/'failed.log', root/'failed.json', interval=.01, stall_seconds=10)
            self.assertEqual(read(root/'failed.json')['exit_code'], 7)

    def test_campaign_lock_prevents_second_live_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            with campaign_lock(Path(tmp)):
                with self.assertRaises(OSError):
                    with campaign_lock(Path(tmp)):
                        self.fail('不应得到第二个锁')

    def test_surviving_child_blocks_restart_without_killing_it(self):
        import os
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root/'processes/child.json', dict(status='monitor_interrupted',
                pid=os.getpid(), command=psutil.Process().cmdline()))
            with self.assertRaisesRegex(RuntimeError, '仍在运行'):
                reject_live_children(root)
            self.assertTrue(psutil.pid_exists(os.getpid()))

    def test_remaining_training_uses_global_target_not_additional_fifteen_million(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = runner.resolve(runner.parser().parse_args(['run', '--output-dir', str(root)]))
            record = dict(seed=42, experiment_name='single_frame_mlp_15m_1',
                model_path=str(root/'final.zip'), log_dir=str(root/'log'), checkpoint_dir=str(root/'checkpoints'),
                tensorboard_dir=str(root/'tb'), tensorboard_run_dir=str(root/'tb/run'))
            Path(record['checkpoint_dir']).mkdir()
            checkpoint = Path(record['checkpoint_dir'])/'single_frame_mlp_15m_1_10002432_steps.zip'
            checkpoint.touch()
            atomic_json(root/'mlp_training.json', record)
            from scripts import train_mobile_whisker_ppo as trainer
            observed = []
            with patch.object(runner, 'model_header', return_value={'evaluated_checkpoint_steps':10_002_432}), \
                 patch.object(runner, 'validate_mlp', return_value={'sha256':'final','evaluated_checkpoint_steps':15_003_648}), \
                 patch.object(trainer, 'train', side_effect=lambda value: observed.append(value)):
                runner.train_mlp(args, {'training_config':{}})
            self.assertEqual(observed[0].timesteps, 4_997_568)
            self.assertEqual(observed[0].resume_from, checkpoint)
            self.assertEqual(observed[0].temporal_encoder, 'mlp')
            self.assertEqual(observed[0].history_length, 1)

    def test_real_mlp_checkpoint_recovers_after_interruption(self):
        import yaml
        from stable_baselines3 import PPO
        from scripts import train_mobile_whisker_ppo as trainer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = runner.resolve(runner.parser().parse_args(['run', '--output-dir', str(root),
                '--mlp-timesteps', '128', '--mlp-n-envs', '1', '--mlp-n-steps', '32',
                '--mlp-checkpoint-freq', '32', '--seed', '7']))
            cfg = dict(plume_model='dynamic', world_bounds=[-2.5,2.5], max_steps=8,
                domain_randomization=False, plume_overrides={'puff_warmup_steps':2,'max_puffs':30})
            (root/'mlp_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
            original = PPO.learn
            def interrupt(model, *a, **kw):
                kw['total_timesteps'] = 32
                original(model, *a, **kw)
                # 模拟工作进程退出时释放的文件句柄，便于Windows同进程测试清理。
                model.get_env().close()
                model.logger.close()
                raise KeyboardInterrupt('人工测试：首个checkpoint后中断')
            with patch.object(trainer, 'ROOT', root/'training_root'):
                with patch.object(PPO, 'learn', interrupt), self.assertRaises(KeyboardInterrupt):
                    runner.train_mlp(args, {'training_config':cfg})
                record = read(root/'mlp_training.json')
                checkpoint = next(Path(record['checkpoint_dir']).glob('*_32_steps.zip'))
                original_sha = __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest()
                runner.train_mlp(args, {'training_config':cfg})
                final = read(root/'mlp_training.json')
                self.assertEqual(final['actual_timesteps'], 128)
                self.assertEqual(final['status'], 'completed')
                self.assertEqual(final['experiment_name'], record['experiment_name'])
                self.assertEqual(__import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest(), original_sha)


if __name__ == '__main__':
    unittest.main()
