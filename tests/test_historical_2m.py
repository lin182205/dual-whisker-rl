"""新增历史主方法只改展示标识；训练窗口退出必须恢复父派发器。"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from dual_whisker_rl.arrival_protocol import atomic_json
from scripts import reevaluate_historical_2m as runner


class HistoricalComparisonTests(unittest.TestCase):
    def test_add_old_main_without_relabeling_primary_or_changing_policy(self):
        base=dict(models=[dict(variant='main')]+[dict(variant=n) for n in ('fixed','periodic','single_frame','history_mlp','no_dr')],labels={},sampling={})
        model=dict(variant='main',variant_config=dict(method='main',config={'domain_randomization':True}),sha256='sha')
        args=SimpleNamespace(reuse_from=Path('source'),model_path=Path('primary'),sealed_source=Path('sealed'),eval_seed=1,eval_repeats=1)
        with patch.object(runner,'comparison_spec',return_value=copy.deepcopy(base)),patch.object(runner,'read',return_value={'settings':{'seeds':[1,2,3,4,5],'timesteps':2_000_000}}),patch.object(runner,'native_model',side_effect=lambda *a:copy.deepcopy(model)):
            spec=runner.build_spec(args)
        self.assertEqual(spec['models'][0]['variant'],'main');self.assertEqual(len(spec['models']),11)
        self.assertEqual([m['variant'] for m in spec['models'][1:6]],['historical_main_2m']*5)
        self.assertTrue(all(m['variant_config']==model['variant_config'] for m in spec['models'][1:6]))
        self.assertEqual(model['variant'],'main')

    def test_display_rename_does_not_change_shared_evaluation_key(self):
        from dual_whisker_rl import stochastic_protocol as protocol
        model=dict(variant='main',sha256='model',variant_config={'method':'main','config':{'domain_randomization':True}})
        spec=dict(models=[model],sampling={'protocol':'stochastic-arrival-v2','eval_repeats':1},config={},
            scenarios={'test':[{'scenario_id':'scene','seed':1}]},scientific_code_sha256={'science':'sha'})
        with patch.object(protocol,'effective_environment',return_value=({'domain_randomization':False},{'physics':'same'})):
            before=protocol.make_jobs(spec,Path('store'))[0]
            changed=copy.deepcopy(spec);changed['models'][0]['variant']='historical_main_2m'
            after=protocol.make_jobs(changed,Path('store'))[0]
        self.assertEqual(before['signature'],after['signature']);self.assertEqual(before['output'],after['output'])

    def test_waits_for_inflight_task_before_opening_evaluation_window(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'new';out.mkdir();old=root/'old';old.mkdir();(root/'results').mkdir()
            atomic_json(root/'results/active_formal_campaign.json',dict(pid=10,created=20.,output_dir=str(old)))
            atomic_json(old/'deployment.json',dict(command=['python','campaign']))
            child=Mock(pid=11);child.status.return_value=psutil.STATUS_RUNNING;child.cmdline.return_value=['python','e4_parallel_worker.py','job.json']
            parent=Mock(pid=10);parent.status.return_value=psutil.STATUS_RUNNING;parent.children.return_value=[child]
            def identity(r):return parent if r['pid']==10 else None
            with patch.object(runner,'ROOT',root),patch.object(runner,'process_from_receipt',side_effect=[parent,child,None,parent]),patch.object(runner,'owned_receipt',side_effect=[{'pid':10},{'pid':11}]),patch.object(runner.time,'sleep') as sleep:
                with runner.training_window(SimpleNamespace(output_dir=out,monitor_interval=.01)):pass
            sleep.assert_called_once();parent.resume.assert_called_once()

    def test_resume_parent_even_when_evaluation_fails(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'new';out.mkdir();old=root/'old';old.mkdir();(root/'results').mkdir()
            atomic_json(root/'results/active_formal_campaign.json',dict(pid=10,created=20.,output_dir=str(old)))
            atomic_json(old/'deployment.json',dict(command=['python','campaign']))
            parent=Mock(pid=10);parent.status.return_value=psutil.STATUS_RUNNING;parent.children.return_value=[]
            args=SimpleNamespace(output_dir=out,monitor_interval=.01)
            with patch.object(runner,'ROOT',root),patch.object(runner,'process_from_receipt',return_value=parent),patch.object(runner,'owned_receipt',return_value=dict(pid=10,created=20.,command=['python','campaign'])):
                with self.assertRaisesRegex(RuntimeError,'evaluation failed'):
                    with runner.training_window(args):raise RuntimeError('evaluation failed')
            parent.suspend.assert_called_once();parent.resume.assert_called_once()

    def test_inflight_other_stage_is_not_suspended_indefinitely(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'new';out.mkdir();old=root/'old';old.mkdir();(root/'results').mkdir()
            atomic_json(root/'results/active_formal_campaign.json',dict(pid=10,created=20.,output_dir=str(old)))
            atomic_json(old/'deployment.json',dict(command=['python','campaign']))
            child=Mock();child.status.return_value=psutil.STATUS_RUNNING;child.cmdline.return_value=['python','other_stage.py']
            parent=Mock(pid=10);parent.status.return_value=psutil.STATUS_RUNNING;parent.children.return_value=[child]
            with patch.object(runner,'ROOT',root),patch.object(runner,'process_from_receipt',return_value=parent),patch.object(runner,'owned_receipt',return_value={}):
                with self.assertRaises(RuntimeError):
                    with runner.training_window(SimpleNamespace(output_dir=out,monitor_interval=.01)):pass
            parent.resume.assert_called_once()


if __name__=='__main__':unittest.main()
