"""红字实验的科学口径、复现与数据完整性检查。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import yaml
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.experiment_env import ExperimentMobileEnv,FEATURE_MASKS,make_experiment_env
from dual_whisker_rl.experiment_protocol import hardware_schedule,holm,paired_interval,seal
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sensor_analysis",ROOT/"scripts/analyze_sensor_experiments.py")
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)

def config():
    return dict(world_bounds=[-2.5,2.5],scenario_mode="randomized",max_steps=40,
                wind_speed_range=[.12,.20],plume_overrides={"puff_warmup_steps":5},
                domain_randomization=False)

class ExperimentEnvironmentTests(unittest.TestCase):
    def test_nominal_equivalence(self):
        base = MobileWhiskerPuffEnv(config())
        experiment = ExperimentMobileEnv({**config(),"success_hold_s":0})
        a,_ = base.reset(seed=21)
        b,_ = experiment.reset(seed=21)
        np.testing.assert_array_equal(a,b)
        for i in range(20):
            action = [5,i%10,(i*3)%10]
            left,right = base.step(action),experiment.step(action)
            np.testing.assert_array_equal(left[0],right[0])
            self.assertAlmostEqual(left[1],right[1])
            self.assertEqual(left[2:4],right[2:4])
        base.close()
        experiment.close()

    def test_domain_randomization_default_equivalence(self):
        cfg = {**config(), "domain_randomization": True}
        base = MobileWhiskerPuffEnv(cfg)
        experiment = ExperimentMobileEnv({**cfg, "success_hold_s": 0})
        a, _ = base.reset(seed=18)
        b, _ = experiment.reset(seed=18)
        np.testing.assert_array_equal(a, b)
        for i in range(20):
            action = [5, i % 10, 9 - i % 10]
            left, right = base.step(action), experiment.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertAlmostEqual(left[1], right[1])

    def test_continuous_success_hold(self):
        env = ExperimentMobileEnv({**config(),"success_hold_s":1})
        env.reset(seed=1,options={"scenario":{"source_position":[0,0],"robot_pose":[0,0,0]}})
        for i in range(5):
            _,_,terminated,_,info = env.step([5,5,5])
            self.assertEqual(terminated,i==4)
            self.assertEqual(info["is_success"],i==4)

    def test_goal_hold_starts_at_first_detected_arrival(self):
        env = ExperimentMobileEnv({**config(), "success_hold_s": 1})
        env.reset(seed=1, options={"scenario": {"source_position": [0, 0], "robot_pose": [0.115, 0, float(np.pi)]}})
        _, _, terminated, _, info = env.step([0, 5, 5])
        self.assertFalse(terminated)
        self.assertEqual(info["success_hold_s"], 0)
        for i in range(5):
            _, _, terminated, _, info = env.step([5, 5, 5])
            self.assertEqual(terminated, i == 4)

    def test_feature_masks_and_bounds(self):
        for name,indices in FEATURE_MASKS.items():
            env = ExperimentMobileEnv({**config(),"feature_mask":name})
            obs,_ = env.reset(seed=33)
            self.assertTrue(env.observation_space.contains(obs),name)
            self.assertTrue(np.all(obs[indices]==0),name)
            obs,*_ = env.step([5,2,7])
            self.assertTrue(np.all(obs[indices]==0),name)

    def test_fixed_reset_builds_one_frame(self):
        env = make_experiment_env(config(),dict(method="b2_fixed",history_length=5,fixed_sector=4),seed=1)
        builder = env.unwrapped.observation_builder
        builder.build = Mock(wraps=builder.build)
        obs,_ = env.reset(seed=4)
        self.assertEqual(builder.build.call_count,1)
        self.assertEqual(obs.shape,(80,))
        self.assertEqual(env.unwrapped.last_left_sector,4)

    def test_fault_replay(self):
        fault = dict(left_gain=.8,sample_drop_probability=.1,action_delay_steps=2,wind_turn_deg=30,start_s=.4)
        env = ExperimentMobileEnv({**config(),"fault":fault})
        def replay():
            obs,_ = env.reset(seed=22)
            result = [obs.tolist()]
            for i in range(15):
                obs,reward,*_ = env.step([5,i%10,9-i%10])
                result.append([obs.tolist(),reward,env.plume.wind_direction])
            return result
        self.assertEqual(replay(),replay())

    def test_all_configured_variants_construct(self):
        suite = yaml.safe_load((ROOT/"configs/red_experiments.yaml").read_text(encoding="utf-8"))
        for name,variant in suite["variants"].items():
            env = make_experiment_env(config(),variant,seed=1)
            obs,_ = env.reset(seed=3)
            self.assertTrue(env.observation_space.contains(obs),name)
            action = env.action_space.sample()
            obs,*_ = env.step(action)
            self.assertTrue(env.observation_space.contains(obs),name)
            env.close()

    def test_unknown_fault_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentMobileEnv({**config(),"fault":{"imaginary_device":1}})

class ProtocolTests(unittest.TestCase):
    def test_schedule_counts_and_pairing(self):
        self.assertEqual(len(hardware_schedule("static")),108)
        self.assertEqual(len(hardware_schedule("drift")),9)
        rows = hardware_schedule("real")
        self.assertEqual(len(rows),360)
        blocks = {r["block_id"] for r in rows}
        self.assertEqual(len(blocks),90)
        for block in blocks:
            items = [r for r in rows if r["block_id"]==block]
            self.assertEqual({r["order"] for r in items},{1,2,3,4})
            self.assertEqual(len({r["start_index"] for r in items}),1)
        self.assertEqual(sum(r["training_seed"]=="" for r in rows),90)
        self.assertEqual(len({r["date"] for r in hardware_schedule("real", repeats=20)}),3)

    def test_scene_tampering_rejected(self):
        spec = importlib.util.spec_from_file_location("red_runner", ROOT / "scripts/run_red_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        args = runner.parser().parse_args(["prepare", "--variants", "main"])
        _, profile, cfg, _, _, payload = runner.resolve(args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.prepare(root, profile, cfg, payload)
            scene_file = root / "scenarios_test.json"
            items = json.loads(scene_file.read_text(encoding="utf-8"))
            items[0]["wind_speed"] += .1
            scene_file.write_text(json.dumps(items), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.prepare(root, profile, cfg, payload)
            self.assertEqual(json.loads(scene_file.read_text(encoding="utf-8")), items)

    def test_manifest_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            self.assertEqual(seal(path,{"seed":1}),seal(path,{"seed":1}))
            with self.assertRaises(ValueError):
                seal(path,{"seed":2})

    def test_seed_not_episode_significance(self):
        a = {(s,str(c)):1.0 for s in range(5) for c in range(100)}
        b = {key:0.0 for key in a}
        result = paired_interval(a,b,draws=100)
        self.assertEqual(result["training_seeds"],5)
        self.assertEqual(result["seed_sign_permutation_p"],.0625)
        self.assertEqual(result["ci95"],[1,1])
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])

    def test_missing_events_not_zero(self):
        self.assertEqual(paired_interval({}, {})["status"],"unpaired")

class SensorAnalysisTests(unittest.TestCase):
    def test_static_day_holdout(self):
        rows = []
        for day in ("2026-10-11","2026-10-12","2026-10-13"):
            for value in (0,50,100,200):
                for t in range(5):
                    rows.append(dict(run_id=day,plateau_id=f"{day}_{value}",date=day,sequence_id=day,
                        direction="ascending",reference_unit="ppm",reference_value=str(value),
                        t_rel_s=str(t),left_adc=str(100+2*value),right_adc=str(120+3*value)))
        platforms,report = sensor.static_analysis(rows,degree=1)
        self.assertEqual(report["holdout_platforms"],4)
        self.assertLess(report["model"]["left"]["holdout_mae"],1e-8)
        self.assertLess(report["model"]["right"]["holdout_mae"],1e-8)
        self.assertEqual(len(platforms),12)

    def test_reference_required(self):
        row = dict(run_id="x",plateau_id="x",date="day",sequence_id="x",direction="ascending",reference_unit="ppm",t_rel_s="0",left_adc="10",right_adc="10")
        with self.assertRaises(ValueError):
            sensor.static_analysis([row])

    def test_drift_exact_slope(self):
        rows = [dict(run_id="x",t_rel_s=str(t),left_adc=str(100+t),right_adc=str(200+2*t)) for t in range(50)]
        report,features = sensor.drift_analysis(rows,draws=20)
        self.assertAlmostEqual(report[0]["left"]["drift_adc_per_hour"],3600)
        self.assertAlmostEqual(report[0]["right"]["drift_adc_per_hour"],7200)
        self.assertEqual(len(features),150)

    def test_early_failure_penalized(self):
        rows = [dict(run_id="x",method="main",training_seed="1",scenario_id="x",
                     success="0",termination="out_of_bounds",elapsed_s="2")]
        converted,_ = sensor.analyze_trials(rows,80)
        self.assertEqual(converted[0]["penalized_time_s"],80)

if __name__=="__main__":
    unittest.main()
