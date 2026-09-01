import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from oht_routing.utils.environment_capture import ActorEnvironmentCapture


def rail(rail_id, oht_ids):
    return SimpleNamespace(
        ID=rail_id,
        SimID=rail_id + 100,
        Distance=100.0,
        DistancePerVelocity=2.0,
        LineType=0,
        PortCount=1,
        LevelJoiningLineIDList=[],
        Level2JoiningLineIDList=[],
        Level3JoiningLineIDList=[],
        DivergingLineIDList=[],
        OhtList=list(oht_ids),
        PredictedOHTIDList=list(oht_ids),
        PredictedOHTCount=len(oht_ids),
        SimAvgSpeed=25.0,
        IdleOHTCount=0,
        ReservationPortCount=0,
    )


class EnvironmentCaptureTests(unittest.TestCase):
    def test_capture_writes_full_snapshots_and_tracks_rail_dwell(self):
        current_pass = SimpleNamespace(
            ID=10, State=2, PassTime=3.0, Distance=20.0
        )
        oht = SimpleNamespace(
            ID=101,
            Name="OHT-101",
            State=2,
            CurrentDistance=20.0,
            JobID=501,
            DestinationLine=20,
            RouteList=[10, 20],
            IdleTime=0.0,
            StopTime=4.0,
            NotPassTimes=[current_pass],
            PassTimes=[],
            FrontOhts={10: [SimpleNamespace(ID=102, Distance=15.0)]},
            VelByLine={10: 10.0},
            RemainTime=8.0,
            RemainingDistanace=80.0,
            OperationRate=75.0,
            OperationTAT=750.0,
            OhtIndividualTat=12.0,
            DispatchedCommand=501,
            RunningAreaType=1,
            CarrierTypes=[2],
            PassDistance=20.0,
            CmdCompleteTat={},
        )
        job = SimpleNamespace(
            ID=501,
            State=3,
            Priority=7,
            OHTId=101,
            FromNode=1,
            ToNode=2,
            IsEqp=1,
            ReAssignCount=2,
            RouteList=[10, 20],
            Waiting_PassLines=[],
            Transfer_PassLines=[],
            CarrierTypes=[2],
            RunningAreaTyes=[1],
        )
        pclient = SimpleNamespace(
            SimTime=10.0,
            TotalTat=160.0,
            TotalOhtOperationRate=0.75,
            CompletedCommandCount=5,
            TransferCommandCount=1,
            WaitingCommandCount=2,
            QueuedCommandCount=3,
            RAILLINE_DIC={10: rail(10, [101]), 20: rail(20, [])},
            RAILLINECOST_DIC={
                10: SimpleNamespace(FRailLineCost=1.2),
                20: SimpleNamespace(FRailLineCost=0.9),
            },
            OHT_DIC={101: oht},
            JOB_DIC={501: job},
        )
        topology = SimpleNamespace(
            all_rail_ids=[10, 20],
            controlled_rail_ids=[10],
            topology_hash="topology-hash",
            mapping_hash="mapping-hash",
        )
        client = SimpleNamespace(
            topology=topology,
            _current_baseline=[1.0, 0.8],
            last_policy_action=[0.4],
            last_applied_action=[0.2],
            parameterDw={10: 2.5},
            parameterC={10: 0.5},
            last_diagnostics={
                "env/step": 400_001.0,
                "env/episode": 8.0,
                "episode/step": 1.0,
                "reward/total_mean": -0.25,
                "rl/action_mean": 0.4,
                "rail_cost/residual_mean": 0.2,
                "critic/q1_mean": 100.0,
            },
            total_steps=400_001,
            episode_id=8,
            episode_steps=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            capture = ActorEnvironmentCapture(
                directory,
                max_steps=2,
                flush_interval=1,
                timestamp="20260821_120000",
            )
            self.assertTrue(capture.capture(pclient, client))
            pclient.SimTime = 12.0
            current_pass.PassTime = 5.0
            oht.CurrentDistance = 40.0
            client.last_diagnostics["env/step"] = 400_002.0
            self.assertTrue(capture.capture(pclient, client))
            self.assertTrue(capture.completed)

            with capture.manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(manifest["rail_count"], 2)
            self.assertEqual(manifest["start_global_step"], 400_001)
            self.assertEqual(manifest["last_global_step"], 400_002)
            self.assertGreater(manifest["file_sizes_bytes"]["steps"], 0)

            with gzip.open(capture.topology_path, "rt", encoding="utf-8") as stream:
                topology_record = json.load(stream)
            self.assertEqual(topology_record["topology_hash"], "topology-hash")
            self.assertTrue(topology_record["rails"][0]["controlled"])

            with gzip.open(capture.steps_path, "rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(len(records), 2)
            first_oht = records[0]["ohts"][0]
            second_oht = records[1]["ohts"][0]
            self.assertEqual(first_oht["current_rail_id"], 10)
            self.assertEqual(first_oht["reported_current_rail_time_s"], 3.0)
            self.assertEqual(first_oht["observed_current_rail_dwell_s"], 0.0)
            self.assertTrue(first_oht["observed_dwell_is_lower_bound"])
            self.assertEqual(second_oht["reported_current_rail_time_s"], 5.0)
            self.assertEqual(second_oht["observed_current_rail_dwell_s"], 2.0)
            self.assertEqual(records[0]["rails"][0]["oht_ids"], [101])
            self.assertEqual(records[0]["rails"][0]["policy_action"], 0.4)
            self.assertEqual(records[0]["jobs"][0]["job_id"], 501)
            self.assertIn("reward/total_mean", records[0]["diagnostics"])
            self.assertIn("rl/action_mean", records[0]["diagnostics"])
            self.assertIn(
                "rail_cost/residual_mean", records[0]["diagnostics"]
            )
            self.assertNotIn("critic/q1_mean", records[0]["diagnostics"])


if __name__ == "__main__":
    unittest.main()
