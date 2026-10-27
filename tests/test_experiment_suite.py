"""验证实验选择、训练共享、协议隔离与真实数据到论文图表的追溯。"""
import csv
import io
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
import unittest
from unittest.mock import Mock, patch
from scripts import run_red_experiments as runner
from scripts import experiment_suite as suite
from scripts.experiment_reports import export_report
from dual_whisker_rl.experiment_protocol import write_json


class ExperimentSelectionTests(unittest.TestCase):
    def resolve(self, flags):
        args=runner.parser().parse_args(flags)
        return args,runner.resolve(args)

    def test_default_is_all_and_training_is_deduplicated(self):
        args,(spec,profile,_,names,_,_)=self.resolve(["plan","--profile","formal"])
        plan=suite.suite_plan(args,spec,profile,names)
        self.assertEqual(plan["selected_experiments"],["e3","e4","e5","e6"])
        self.assertEqual(plan["native_training_tasks"],165)
        self.assertEqual(plan["e4_training_tasks"],35)
        self.assertEqual(plan["training_tasks"],200)
        self.assertEqual(runner.parser().parse_args([]).phase,"run")

    def test_subset_alias_and_nominal_are_resolved_before_writing(self):
        args,_=self.resolve(["plan","--experiment","E6,E3","--faults","wind24"])
        self.assertEqual(args.experiments,["e3","e6"])
        self.assertEqual(args.faults,"nominal,wind24")
        for value in ["e2","e3,e3","all,e3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(["plan","--experiments",value])

    def test_phase_alias_rejects_conflicting_experiment_selection(self):
        with self.assertRaises(ValueError):
            self.resolve(["robustness","--experiments","e3"])
        args,_=self.resolve(["robustness"])
        self.assertEqual(args.experiments,["e6"])

    def test_shared_native_models_train_once_and_only_selected_tests_run(self):
        args,(spec,profile,cfg,names,_,_)=self.resolve(["run","--experiments","e3","e5","e6","--variants","main,h1"])
        train,evaluate,summarize=Mock(),Mock(),Mock()
        def report(experiment,root,out,*a,**kw):
            return dict(experiment=experiment,status="complete",report=str(out/"report.md"),figures=1,tables=1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch.object(suite,"export_report",side_effect=report):
            suite.run_suite(args,Path(tmp),names,spec,profile,cfg,dict(signature="test"),
                            train_one=train,evaluate=evaluate,summarize=summarize)
        self.assertEqual(train.call_count,2)
        self.assertEqual(evaluate.call_count,2)
        self.assertNotIn("robust",evaluate.call_args_list[0].kwargs)
        self.assertTrue(evaluate.call_args_list[1].kwargs["robust"])

    def test_e4_dispatches_original_protocol_without_native_training(self):
        args,(spec,profile,cfg,names,_,_)=self.resolve(["run","--experiments","e4","--timesteps","64","--e4-methods","main,b0_random"])
        train=Mock()
        def report(experiment,root,out,*a,**kw):
            return dict(experiment=experiment,status="complete",report=str(out/"report.md"),figures=0,tables=1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch.object(suite,"export_report",side_effect=report), patch.object(suite.subprocess,"run") as call:
            suite.run_suite(args,Path(tmp),names,spec,profile,cfg,dict(signature="test"),
                            train_one=train,evaluate=Mock(),summarize=Mock())
        train.assert_not_called()
        command=call.call_args.args[0]
        self.assertIn("main,b0_random",command)
        self.assertIn("64",command)
        self.assertIn("40",command)


class PaperReportTests(unittest.TestCase):
    def fixture(self,root,names=("main","h1"),seeds=(1,2)):
        args=runner.parser().parse_args(["summarize","--bootstrap","30"])
        spec,profile,cfg,_,_,_=runner.resolve(args)
        profile["seeds"]=list(seeds)
        write_json(root/"manifest.json",dict(signature="test"))
        write_json(root/"scenarios_test.json",[dict(id=0),dict(id=1)])
        for n in names:
            for seed in seeds:
                for i in range(2):
                    row=dict(variant=n,training_seed=seed,scenario_id=f"s{i}",fault="nominal",
                        odor_hit_ratio=.2 if n=="main" else .4,success=i==0,
                        whisker_only_reacquisition_events=1,penalized_time_s=3)
                    write_json(root/n/f"seed_{seed}"/"episodes"/"nominal"/f"s{i}.json",row)
        return args,spec,profile,cfg

    def test_paired_effect_source_counts_and_vector_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"input"
            args,spec,profile,_=self.fixture(root)
            out=Path(tmp)/"report"
            receipt=export_report("e5",root,out,args,["main","h1"],spec,profile,dict(signature="test"))
            self.assertEqual(receipt["status"],"complete")
            self.assertEqual(receipt["figures"],1)
            rows=list(csv.DictReader(io.StringIO((out/"tables/paired_effects.csv").read_text(encoding="utf-8-sig"))))
            row=next(r for r in rows if r["metric"]=="odor_hit_ratio")
            self.assertAlmostEqual(float(row["difference"]),.2)
            self.assertEqual(int(row["training_seeds"]),2)
            self.assertEqual(int(row["common_episodes"]),4)
            for suffix in ["png","pdf","svg"]:
                self.assertGreater((out/f"figures/e5_history.{suffix}").stat().st_size,100)
            self.assertIn("<text",(out/"figures/e5_history.svg").read_text(encoding="utf-8"))
            manifest=json.loads((out/"report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["inputs"]),9)

    def test_missing_controls_and_history_have_tables_without_invented_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"input"
            args,spec,profile,_=self.fixture(root,names=("h1",))
            report=export_report("e5",root,Path(tmp)/"e5",args,["h1"],spec,profile,dict(signature="test"))
            self.assertEqual(report["status"],"partial_or_missing_controls")
            self.assertEqual(report["figures"],0)
            report=export_report("e3",root,Path(tmp)/"e3",args,["main"],spec,profile,
                dict(signature="test",external_model=dict(training_seed=1)))
            self.assertEqual(report["status"],"missing_training_history")
            self.assertEqual(report["figures"],0)
            self.assertTrue((Path(tmp)/"e3/tables/experiment_status.csv").exists())

    def test_single_model_fault_intervals_are_conditional_and_zero_ratio_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"input"
            args,spec,profile,_=self.fixture(root,names=("main",),seeds=(1,))
            args.faults="nominal,delay2"
            args.limit=2
            for fault,value in [("nominal",0.0),("delay2",.2)]:
                for i in range(2):
                    write_json(root/"main/seed_1/robustness"/fault/f"s{i}.json",dict(
                        variant="main",training_seed=1,scenario_id=f"s{i}",fault=fault,
                        odor_hit_ratio=value,success=False,whisker_only_reacquisition_events=0,penalized_time_s=3))
            out=Path(tmp)/"e6"
            report=export_report("e6",root,out,args,["main"],spec,profile,dict(signature="test"))
            self.assertEqual(report["status"],"complete")
            rows=list(csv.DictReader(io.StringIO((out/"tables/degradation.csv").read_text(encoding="utf-8-sig"))))
            row=next(r for r in rows if r["metric"]=="odor_hit_ratio")
            self.assertEqual(row["ratio"],"")
            self.assertEqual(row["uncertainty_scope"],"conditional_on_one_model")


if __name__=="__main__":
    unittest.main()
