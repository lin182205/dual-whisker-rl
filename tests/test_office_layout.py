"""办公功能区、公共通道、气团传播与旧地图隔离的回归。"""

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.envs.room_layouts import point_in_polygon
from dual_whisker_rl.envs.office_layout import SMALL_OFFICE
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
CONFIG = yaml.safe_load((ROOT/'configs/mobile_rooms_hybrid.yaml').read_text(encoding='utf-8'))
OLD = yaml.safe_load((ROOT/'configs/mobile_rooms_hybrid_six.yaml').read_text(encoding='utf-8'))
CENTERS = {'R1':(-2.9,-3.4),'R2':(-2.9,-.2),'R3':(-2.9,3.2),'R4':(2.9,-.1),
           'R5':(2.9,2.6),'R6':(2.9,4.2),'reception':(2.1,-3.5),'corridor':(0.,1.4)}


def make_env(cfg=CONFIG):
    cfg = copy.deepcopy(cfg)
    cfg['plume_overrides']['puff_warmup_steps'] = 0
    return MobileWhiskerPuffEnv(cfg)


class OfficeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.map = RoomMap(layout=SMALL_OFFICE)

    def test_complete_coverage_area_semantics_and_ids(self):
        self.assertEqual(CONFIG['map_layout'], SMALL_OFFICE)
        self.assertEqual(self.map.metadata()['room_count'], 6)
        self.assertEqual(self.map.metadata()['region_count'], 8)
        area = 0.
        for name, poly in self.map.regions.items():
            p = np.asarray(poly)
            area += abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2.
            self.assertEqual(self.map.room_id(CENTERS[name]), name)
        self.assertAlmostEqual(area, 100.)
        for x in np.linspace(-4.999,4.999,61):
            for y in np.linspace(-4.997,4.997,61):
                self.assertEqual(sum(point_in_polygon((x,y),poly) for poly in self.map.regions.values()),1)
        self.assertEqual({r['function'] for r in self.map.metadata()['regions']},
                         {'office','meeting_room','pantry','restroom','reception','circulation'})

    def test_all_doors_cross_into_correct_public_areas(self):
        self.assertEqual(len(self.map.doors), 8)
        degree = {name:0 for name in CENTERS}
        for door in self.map.doors:
            c = np.asarray(door['center'])
            n = np.array([1.,0.]) if door['axis']=='vertical' else np.array([0.,1.])
            a,b = c-.4*n,c+.4*n
            self.assertTrue(self.map.visible(a,b,.15),door['id'])
            self.assertEqual({self.map.room_id(a),self.map.room_id(b)},set(door['regions']))
            for name in door['regions']: degree[name] += 1
            if door['kind']=='doorway':
                self.assertEqual(door['width_m'],1.)
                self.assertTrue(any(r in ('corridor','reception') for r in door['regions']))
        for name in ('R1','R2','R3','R5','R6'): self.assertEqual(degree[name],1)
        self.assertEqual(degree['R4'],2)

    def test_hallway_clear_width_and_unobstructed_lobby_transition(self):
        ranges = self.map.raycast((0.,1.),0.,360,3.)
        self.assertAlmostEqual(float(ranges[0]+ranges[180]),1.5,places=6)
        self.assertTrue(self.map.visible((0.,-4.5),(0.,4.5),.15))
        self.assertFalse(self.map.visible((-2.,0.),(2.,0.),.15))
        self.assertTrue(self.map.visible((3.4,-3.),(3.4,-1.),.15))
        self.assertFalse(self.map.visible((2.,-3.),(2.,-1.),.15))

    def test_paths_between_all_regions_are_finite_symmetric(self):
        vertices,_ = self.map.navigation_graph(.15)
        self.assertGreater(len(vertices),20)
        for a in CENTERS.values():
            for b in CENTERS.values():
                d = self.map.path_distance(a,b,.15)
                self.assertTrue(math.isfinite(d))
                self.assertGreaterEqual(d+1e-8,math.dist(a,b))
                self.assertAlmostEqual(d,self.map.path_distance(b,a,.15),places=6)

    def test_legacy_hashes_and_geometry_remain_exact(self):
        self.assertEqual(RoomMap(layout='irregular_six_rooms_v1').digest,
                         'eb24d3c3f555ed93ba25ae80177151ffa49aeddce90f31049d6e200b0aeac5db')
        self.assertEqual(RoomMap().digest,'a9a9ab9dcb55d7ee4dbfcc85ab976135e0a13436ba7283638ca6d60c297c7ba4')
        self.assertNotEqual(self.map.digest,RoomMap(layout='irregular_six_rooms_v1').digest)


class OfficeEnvironmentTests(unittest.TestCase):
    def test_random_initialization_covers_all_regions(self):
        env = make_env()
        self.addCleanup(env.close)
        starts,sources = set(),set()
        for seed in range(256):
            obs,info = env.reset(seed=seed)
            self.assertEqual(obs.shape,(57,))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(env.room_navigation.map.valid(info['source'],.25))
            self.assertTrue(env.room_navigation.map.valid((env.robot_state.x,env.robot_state.y),.15))
            self.assertGreaterEqual(info['path_distance_to_source'],1.)
            starts.add(info['start_room']);sources.add(info['source_room'])
        self.assertEqual(starts,set(CENTERS));self.assertEqual(sources,set(CENTERS))

    def test_continuous_door_passage_and_wall_collision(self):
        env = make_env()
        self.addCleanup(env.close)
        env.reset(seed=7)
        nav = env.room_navigation
        from dual_whisker_rl.envs.robot_model import RobotState
        self.assertGreater(nav.move_arc(RobotState(-1.3,-3.5,0.),.15,0.,8.).x,-.8)
        self.assertFalse(nav.collision)
        p=nav.move_arc(RobotState(-1.3,-2.5,0.),.15,.01,8.)
        self.assertTrue(nav.collision);self.assertEqual(p.x,-1.3)

    def test_puffs_cross_office_door_and_closed_wall_blocks_concentration(self):
        env = make_env()
        self.addCleanup(env.close)
        env.reset(seed=7)
        plume=env.plume
        def puff(x,y):
            return dict(x=x,y=y,mass=.02,sigma_downwind=.4,sigma_crosswind=.4,
                        cos_direction=1.,sin_direction=0.,direction=0.,age=0.,vd=0.,vc=0.)
        plume.puffs=[puff(-1.1,-3.5),puff(-1.1,-2.5)]
        plume.dt,plume.wind_speed,plume.wind_direction=1.,.5,0.
        plume.turbulence_vel_std_downwind=plume.turbulence_vel_std_crosswind=0.
        plume._advance_puffs()
        self.assertGreater(plume.puffs[0]['x'],-.8)
        self.assertLess(plume.puffs[1]['x'],-.85)
        plume.puffs=[puff(-1.1,-2.5)]
        self.assertEqual(plume.concentration(-.4,-2.5,False),plume.gas_background)
        plume.puffs=[puff(-1.1,-3.5)]
        self.assertGreater(plume.concentration(-.4,-3.5,False),plume.gas_background)

    def test_old_hybrid_model_and_scenario_bank_are_rejected(self):
        from stable_baselines3 import PPO
        from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
        from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor
        from train_whisker_only_ppo import ObservationHistoryWrapper
        from evaluate_mobile_fixed_scenarios import prepare_scenario_bank
        old,new=make_env(OLD),make_env()
        self.addCleanup(old.close);self.addCleanup(new.close)
        wrapped=ObservationHistoryWrapper(old,2)
        model=PPO(HybridActorCriticPolicy,wrapped,n_steps=8,batch_size=8,
                  policy_kwargs={'features_extractor_class':OlfactoryNavigationExtractor,
                                 'features_extractor_kwargs':{'history_length':2}})
        model.mobile_environment_contract=environment_contract(old,2)
        model.mobile_environment_contract['encoder_config']=model.policy.features_extractor.configuration
        with self.assertRaisesRegex(ValueError,'契约'):validate_model_environment(model,new)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bank.json'
            prepare_scenario_bank(path,old.effective_config,count=1,base_seed=7,regenerate=False)
            with self.assertRaises(ValueError):
                prepare_scenario_bank(path,new.effective_config,count=1,base_seed=7,regenerate=False)


if __name__=='__main__':unittest.main()
