// Model setup follows the installed MainFrame.LoadSimModel order. No forms or shapes are created.
using System;
using System.Diagnostics;
using Simulation.Model;
using Pinokio.CAP.Data;
using Pinokio.CAP.Manager;
using Pinokio.Sim.Interface;
internal static class HeadlessLoader {
 public static Fab Load(string filePath) {
  TotalSimModel totalSimModel;
  if (!SimModelDBManager.Instance.DownloadSimModels(filePath, out totalSimModel)) throw new InvalidOperationException("Input DB load failed");
  if (!ModelManager.Instance.SetSimulationModelTotalModel(totalSimModel)) throw new InvalidOperationException("Model construction failed");
  Fab createdFab;
  createdFab = ModelManager.Instance.Fabs[totalSimModel.FabName];
  createdFab.TotalSimModel = totalSimModel;
  Stopwatch stopwatch2 = new Stopwatch();
  stopwatch2.Start();
  ModelManager.Instance.Fabs[totalSimModel.FabName].FabAccess_Path = filePath;
  ModelManager.Instance.SetDataToFab(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetDataToFab:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.CreateNetworkLineStructure(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("CreateNetworkLineStructure:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.SetPortLineInZone(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetPortLineInZone:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.SetStopNResetLine(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetStopNResetLine:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.SetRailLineSpec(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetRailLineSpec:   ", stopwatch2);
  stopwatch2.Restart();
  ZoneHelper.Instance.InitBayInLoopSetting(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("InitBayInLoopSetting:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.UpdateShortestDijkstra(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("UpdateShortestDijkstra:   ", stopwatch2);
  stopwatch2.Restart();
  DefineEnum.LogicType pinokioLogicType = createdFab.PinokioLogicType;
  createdFab.PinokioLogicType = DefineEnum.LogicType.AI1_0;
  PathFinder.Instance.UpdateNetworkStateForLoop(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("UpdateNetworkStateForLoop:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.UpdateRouteState(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("UpdateRouteState:   ", stopwatch2);
  stopwatch2.Restart();
  createdFab.PinokioLogicType = pinokioLogicType;
  ZoneHelper.Instance.InitBumpingPortsAndWaitingPorts(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("InitBumpingPortsAndWaitingPorts:   ", stopwatch2);
  stopwatch2.Restart();
  ZoneHelper.Instance.IntializeBumpingRotation(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("IntializeBumpingRotation:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.SetRemainTimeCurLine(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetRemainTimeCurLine:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.LogConsole("InitializePaths:   ", stopwatch2);
  ModelManager.Instance.AddCommander(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("AddCommander:   ", stopwatch2);
  stopwatch2.Restart();
  ModelManager.Instance.InitAstar(createdFab);
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("InitAstar:   ", stopwatch2);
  stopwatch2.Restart();
  createdFab.AvgRailSpeedNode = (UpdateAvgRailSpeedNode)ModelManager.Instance.AddUpdateAvgRailSpeedNode(createdFab, 600.0);
  createdFab.DijkstraNode = (UpdateDijkstraNode)ModelManager.Instance.AddUpdateDijkstraNode(createdFab, 5.0);
  createdFab.RerouteNode = (UpdateRerouteNode)ModelManager.Instance.AddUpdateRerouteNode(createdFab, totalSimModel.specSimModel.ReroutingInterval_Indicator);
  createdFab.RerouteNode.isUsed = false;
  createdFab.UpdateEqpNode = (UpdateEqpNode)ModelManager.Instance.AddUpdateEqpNode(createdFab, totalSimModel.specSimModel.ReroutingInterval_Indicator);
  createdFab.UpdateEqpNode.isUsed = false;
  createdFab.AreabalanceNode = (UpdateAreabalanceNode)ModelManager.Instance.AddUpdateAreabalanceNode(createdFab, 5.0);
  createdFab.IdleOHTBalanceNode = (UpdateIdleOHTBalance)ModelManager.Instance.AddUpdateBalanceNode(createdFab, totalSimModel.specSimModel.IdleBalanceW1);
  createdFab.StandByNode = (UpdateStandByNode)ModelManager.Instance.AddUpdateStandByNode(createdFab, 5.0);
  createdFab.SendSimInfoNode = (UpdateSendSimInfoNode)ModelManager.Instance.AddSendSimInfoNode(createdFab, 1.0, false);
  createdFab.LineAvgSpeedCheck = (UpdateLineAvgSpeedCheck)ModelManager.Instance.AddLineAvgSpeedCheck(createdFab, 10.0, isUsed: true);
  if (ModelManager.Instance.UpdateActivateDeadNode == null)
  {
   ModelManager.Instance.UpdateActivateDeadNode = (UpdateActivateDeadNode)ModelManager.Instance.AddUpdateActivateDeadNode(createdFab, 1.0);
  }
  if (ModelManager.Instance.AccelerationNode == null)
  {
   ModelManager.Instance.AccelerationNode = (UpdateAccelerationTimeNode)ModelManager.Instance.AddUpdateAccelerationNode(createdFab, 30.0);
  }
  if (ModelManager.Instance.UpdateActivateStopOHTNode == null)
  {
   ModelManager.Instance.UpdateActivateStopOHTNode = (UpdateActivateStopOHTNode)ModelManager.Instance.AddUpdateActivateStopOHTNode(createdFab, 1.0);
  }
  if (ModelManager.Instance.SimTimeNode == null)
  {
   ModelManager.Instance.SimTimeNode = (UpdateSimTimeNode)ModelManager.Instance.AddUpdateSimTimeNode(createdFab, 10.0);
  }
  if (ModelManager.Instance.CheckSimTimeNode == null)
  {
   ModelManager.Instance.CheckSimTimeNode = (UpdateCheckSimTimeNode)ModelManager.Instance.AddUpdateCheckSimTimeNode(createdFab, 0.1);
   ModelManager.Instance.CheckSimTimeNode.isUsed = false;
  }
  if (ModelManager.Instance.UpdateAnimationNode == null)
  {
   ModelManager.Instance.UpdateAnimationNode = (UpdateAnimationNode)ModelManager.Instance.AddUpdateAnimationTimeNode(createdFab, 0.02);
   ModelManager.Instance.UpdateAnimationNode.isUsed = false;
  }
  if (ModelManager.Instance.UpdateResultSaveNode == null)
  {
   ModelManager.Instance.UpdateResultSaveNode = (UpdateResultSaveNode)ModelManager.Instance.AddUpdateResultSaveNode(createdFab, 60.0);
  }
  if (ModelManager.Instance.UpdateSummaryNode == null)
  {
   ModelManager.Instance.UpdateSummaryNode = (UpdateSummaryNode)ModelManager.Instance.AddUpdateSummaryNode(createdFab, 300.0);
  }
  if (ModelManager.Instance.AccumulateSummaryNode == null)
  {
   ModelManager.Instance.AccumulateSummaryNode = (UpdateAccumulateSummaryNode)ModelManager.Instance.AddAccumulateSummaryNode(createdFab, 1.0);
  }
  createdFab.PlaybackNode = (UpdatePlaybackLogNode)ModelManager.Instance.AddUpdatePlaybackLogNode(createdFab, createdFab.PlaybackLogTimeInterval);
  createdFab.PlaybackNode.isUsed = false;
  if (ModelManager.Instance.SettingEndNode == null)
  {
   ModelManager.Instance.SettingEndNode = (SettingEndNode)ModelManager.Instance.AddSettingEndNode(createdFab);
  }
  if (ModelManager.Instance.SettingAI20Node == null)
  {
   ModelManager.Instance.SettingAI20Node = (SettingAI20Node)ModelManager.Instance.AddSettingAI20Node(createdFab);
  }
  if (ModelManager.Instance.SettingPauseNode == null)
  {
   ModelManager.Instance.SettingPauseNode = (SettingPauseNode)ModelManager.Instance.AddSettingPauseNode(createdFab);
  }
  if (ModelManager.Instance.LineCostTypeNode == null)
  {
   ModelManager.Instance.LineCostTypeNode = ModelManager.Instance.AddLineCostTypeNode(createdFab);
  }
  if (ModelManager.Instance.SaveNode == null)
  {
   ModelManager.Instance.SaveNode = ModelManager.Instance.AddSaveNode(createdFab, PinokioDataMart.Instance.SavingInterval);
  }
  if (PinokioDataMart.Instance.Pinokio_Save_AIS_DATA)
  {
   createdFab.SaveAISNode = ModelManager.Instance.AddSaveAISNode(createdFab, 0.1);
  }
  if (ModelManager.Instance.OnReinforcement)
  {
   createdFab.ReinforementAI20TrainNode = ModelManager.Instance.AddReinforcementAI20Node_3(createdFab, 5.0, createdFab.RailLines);
  }
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("SetFabSimModels:   ", stopwatch2);
  stopwatch2.Restart();
  PinokioDataMart.Instance.InitializePaths();
  stopwatch2.Stop();
  ModelManager.Instance.LogConsole("InitializePaths:   ", stopwatch2);
  stopwatch2.Restart();
 ModelManager.Instance.CheckZCUStopReset(createdFab);
 return createdFab;
 }
}
