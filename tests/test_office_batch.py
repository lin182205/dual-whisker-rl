"""办公批量对比的动作、墙体物理、消融索引、场景与模型隔离。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.experiment_env import make_experiment_env,ExperimentMobileEnv
from dual_whisker_rl.envs.room_plume import RoomPuffPlume
from dual_whisker_rl.e4_experiments import build_scenarios,RandomController,ReactiveController,make_env
from dual_whisker_rl.agents.hybrid_policy import HybridActionDistribution
from dual_whisker_rl.batch_mobile import policy_setup,attach_contract,validate_contract
from scripts import run_red_experiments as red
from scripts import run_e4_experiments as e4
from scripts.experiment_suite import legacy_command

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'configs/mobile_rooms_hybrid.yaml').read_text(encoding='utf-8'))
torch.set_num_threads(1)

def cfg():
    c=copy.deepcopy(CFG);c['plume_overrides']['puff_warmup_steps']=0
    c['max_steps']=8;c['success_hold_s']=0
    return c


class OfficeBatchTests(unittest.TestCase):
    def test_defaults_and_e4_delegation_use_office(self):
        args=red.parser().parse_args(['plan','--experiments','e4'])
        spec,profile,c,_,_,payload=red.resolve(args)
        self.assertEqual(c['map_layout'],'small_office_v1')
        self.assertEqual(profile['max_steps'],1200)
        self.assertIn('office_experiments',args.output_dir)
        self.assertEqual(c['success_hold_s'],0)
        self.assertEqual(payload['environment_contract']['observation_version'],'mobile_hybrid_nav_v1')
        command=legacy_command(args,ROOT/'results/unused',profile,'prepare')
        self.assertEqual(Path(command[command.index('--config')+1]).name,'mobile_rooms_hybrid.yaml')
        a=e4.build_parser().parse_args(['prepare']);e4.resolve_args(a)
        self.assertEqual(a.max_steps,1200);self.assertIn('office_e4',str(a.output_dir))

    def test_office_nominal_matches_base_physics(self):
        c=cfg();c.pop('success_hold_s')
        a,b=MobileWhiskerPuffEnv(c),ExperimentMobileEnv({**c,'success_hold_s':0})
        self.addCleanup(a.close);self.addCleanup(b.close)
        x,_=a.reset(seed=7);y,_=b.reset(seed=7);np.testing.assert_array_equal(x,y)
        self.assertIsInstance(b.plume,RoomPuffPlume)
        for i in range(8):
            action=np.array([.23,-.17,i,9-i],dtype=np.float32)
            left,right=a.step(action),b.step(action)
            np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:4],right[1:4])

    def test_all_methods_preserve_continuous_motion_and_legal_categories(self):
        for method,n in [('main',4),('b2_fixed',2),('b3_scan',2),('b6_left',3),('b6_right',3)]:
            with self.subTest(method=method):
                env=make_experiment_env(cfg(),{'method':method,'history_length':2,'fixed_sector':4})
                self.addCleanup(env.close);env.reset(seed=7)
                self.assertEqual(env.action_space.shape,(n,))
                action=np.array([.25,.2]+[3.]*(n-2),dtype=np.float32)
                _,_,_,_,info=env.step(action)
                self.assertAlmostEqual(info['requested_linear_velocity'],.09375)
                self.assertAlmostEqual(info['requested_angular_velocity'],.3,places=6)
                if method=='b2_fixed':self.assertEqual((info['left_sector'],info['right_sector']),(4,4))
                if method=='b6_left':self.assertEqual((info['left_sector'],info['right_sector']),(3,4))
                if method=='b6_right':self.assertEqual((info['left_sector'],info['right_sector']),(4,3))

    def test_distribution_with_zero_one_two_categorical_branches(self):
        for n in (0,1,2):
            p=torch.randn(5,2+10*n,requires_grad=True);std=torch.zeros(2,requires_grad=True)
            d=HybridActionDistribution(10,n).proba_distribution(p,std)
            action=d.sample().detach();self.assertEqual(action.shape,(5,2+n))
            (-d.log_prob(action).mean()-.01*d.entropy().mean()).backward()
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad[:,:2].abs().sum().item(),0.)
            for i in range(n):self.assertGreater(p.grad[:,2+10*i:12+10*i].abs().sum().item(),0.)

    def test_fault_delay_holds_true_stop_without_integer_truncation(self):
        env=make_experiment_env(cfg(),{'history_length':1},fault={'action_delay_steps':2})
        self.addCleanup(env.close);env.reset(seed=7)
        for i in range(3):
            _,_,_,_,info=env.step([.25,.2,3,4])
            self.assertAlmostEqual(info['requested_linear_velocity'],0. if i<2 else .09375)

    def test_blank_ablation_masks_age_not_angular_velocity(self):
        env=make_experiment_env(cfg(),{'history_length':1,'feature_mask':'no_blank_age'})
        self.addCleanup(env.close);env.reset(seed=7)
        obs,*_=env.step([-1.,.5,3,4]);self.assertEqual(obs[16],0.)
        self.assertAlmostEqual(float(obs[15]),.5)

    def test_every_office_variant_constructs_with_wall_plume(self):
        spec=yaml.safe_load((ROOT/'configs/office_experiments.yaml').read_text(encoding='utf-8'))
        for name,v in spec['variants'].items():
            with self.subTest(name=name):
                env=make_experiment_env(cfg(),v);obs,_=env.reset(seed=7)
                self.assertEqual(obs.shape,(57*v['history_length'],))
                self.assertIsInstance(env.unwrapped.plume,RoomPuffPlume)
                obs,*_=env.step(env.action_space.sample());self.assertTrue(env.observation_space.contains(obs))
                env.close()

    def test_paired_scenarios_are_valid_replayable_and_not_downwind_restricted(self):
        scenes=build_scenarios(config=cfg(),count=32,seed=11)
        self.assertEqual(scenes,build_scenarios(config=cfg(),count=32,seed=11))
        env=make_env('main',cfg());self.addCleanup(env.close)
        upstream=False;areas=set()
        for scene in scenes:
            _,info=env.reset(seed=scene.seed,options={'scenario':scene.to_dict()})
            self.assertGreaterEqual(info['path_distance_to_source'],1.)
            s=env.unwrapped.robot_state;self.assertEqual((s.x,s.y),scene.robot_pose[:2])
            areas.add(info['source_room'])
            upstream |= np.dot(np.array(scene.robot_pose[:2])-scene.source_position,
                               [np.cos(scene.wind_direction),np.sin(scene.wind_direction)])<0
        self.assertTrue(upstream);self.assertGreater(len(areas),5)

    def test_rules_return_valid_hybrid_actions(self):
        env=make_env('main',cfg());self.addCleanup(env.close);obs,_=env.reset(seed=2)
        for controller in (RandomController(1,hybrid=True),ReactiveController(hybrid=True)):
            for _ in range(8):
                action,_=controller.predict(obs);self.assertTrue(env.action_space.contains(action));env.step(action)

    def test_models_save_load_and_reject_same_dim_different_map(self):
        for method in ('main','b2_fixed','b6_left'):
            variant={'method':method,'history_length':2}
            env=make_experiment_env(cfg(),variant);self.addCleanup(env.close)
            policy,kwargs=policy_setup(env,2)
            model=PPO(policy,env,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs=kwargs,device='cpu')
            attach_contract(model,env,variant);model.learn(8);validate_contract(model,env,variant,resume=True)
            with tempfile.TemporaryDirectory() as tmp:
                model.save(Path(tmp)/'m');loaded=PPO.load(Path(tmp)/'m')
                validate_contract(loaded,env,variant)
                changed=cfg();changed['map_layout']='irregular_six_rooms_v1'
                other=make_experiment_env(changed,variant);self.addCleanup(other.close)
                with self.assertRaisesRegex(ValueError,'契约'):validate_contract(loaded,other,variant)


if __name__=='__main__':unittest.main()
