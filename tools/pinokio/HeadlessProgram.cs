using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Simulation.Engine;
using Simulation.Model;
using Pinokio.CAP.Data;
using Pinokio.CAP.Manager;
using Pinokio.Sim.Definition;
using Pinokio.TCP.IP;
using Pinokio.Utill.Log;

// The entry assembly selects framework compatibility behavior, including the
// ordering of tied List.Sort items used by native idle-vehicle selection.
// A supportedRuntime element in the config alone does not select these semantics.
[assembly: System.Runtime.Versioning.TargetFramework(".NETFramework,Version=v4.8", FrameworkDisplayName=".NET Framework 4.8")]
internal static class HeadlessProgram {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern bool SetDllDirectory(string path);
 [STAThread] static int Main(string[] args) {
  try {
   Console.OutputEncoding=new System.Text.UTF8Encoding(false);
   string framework=AppDomain.CurrentDomain.SetupInformation.TargetFrameworkName;
   if(framework!=".NETFramework,Version=v4.8")throw new InvalidOperationException("GUI runtime compatibility requires .NET Framework 4.8 targeting; got "+framework);
   Console.WriteLine("[headless] framework="+framework);
   var options=new Dictionary<string,string>();
   for(int i=0;i<args.Length;i++) {
    string key=args[i];
    if(key=="--load-only"||key=="--no-python") options[key]="true";
    else {if(!key.StartsWith("--")||i+1==args.Length)throw new ArgumentException("Expected --option value");options[key]=args[++i];}
   }
   foreach(string key in new[]{"--sim-dir","--input","--output-dir","--name"}) if(!options.ContainsKey(key))throw new ArgumentException("Missing "+key);
   string sim=Path.GetFullPath(options["--sim-dir"]);
   SetDllDirectory(sim);
   AppDomain.CurrentDomain.AssemblyResolve+=(s,e)=>{string file=Path.Combine(sim,new AssemblyName(e.Name).Name+".dll");return File.Exists(file)?Assembly.LoadFrom(file):null;};
   return Execute(options);
  } catch(Exception e) {Console.Error.WriteLine("HEADLESS ERROR: "+e);Environment.Exit(1);return 1;}
 }
 [MethodImpl(MethodImplOptions.NoInlining)] static int Execute(Dictionary<string,string> options) {
  string input=Path.GetFullPath(options["--input"]), output=Path.GetFullPath(options["--output-dir"]), name=options["--name"];
  if(!File.Exists(input))throw new FileNotFoundException("Input DB missing",input);
  if(!pLicenseManager.Instance.IsPossibleLicenseCheck())throw new InvalidOperationException("Installed simulator engine license check failed.");
  if(name.IndexOfAny(Path.GetInvalidFileNameChars())>=0||name=="."||name=="..")throw new ArgumentException("Invalid result name");
  Directory.CreateDirectory(output);
  string result=Path.Combine(output,name+".db");
  if(File.Exists(result))throw new IOException("Refusing to overwrite "+result);
  string work=Path.Combine(output,name+"_work");
  if(Directory.Exists(work))throw new IOException("Run work directory already exists: "+work);
  Directory.CreateDirectory(work);
  Environment.CurrentDirectory=work;
  LogManager.Instance.Init(SERVICE_TYPE.SIMULATOR);
  new SimEngine();new SimModelDBManager();new SimResultDBManager();new ModelManager();
  ModelManager.Instance.Initialize();
  new DBConnector();new Scheduler();new PathFinder();new Dispatcher();new ZoneHelper();new WipHelper();
  var manager=ModelManager.Instance;var engine=SimEngine.Instance;
  string model=Path.Combine(work,"input.db");
  using(var stream=File.OpenRead(input)) {byte[] header=new byte[16];stream.Read(header,0,16);
   if(System.Text.Encoding.ASCII.GetString(header).StartsWith("SQLite format 3"))throw new ArgumentException("Input must be a simulator-exported encrypted DB (Commander expects the original export).");
   else LogManager.DecryptFile(input,model);
  }
  Console.WriteLine("[headless] loading "+input);var elapsed=Stopwatch.StartNew();
  Fab fab=HeadlessLoader.Load(model);
  Console.WriteLine("[headless] input model start="+fab.StartDateTime.ToString("yyyy-MM-dd HH:mm:ss")+" end="+fab.EndDateTime.ToString("yyyy-MM-dd HH:mm:ss")+" duration_s="+(fab.EndDateTime-fab.StartDateTime).TotalSeconds);
  // Match the GUI's default Euclidean distance with its penalty toggle off.
  manager.AstarPara.IsUsePenalty=false;
  fab.FabAccess_Path=input;
  manager.SimTimeNode.SimTimeUpdate+=(s,e)=>{};fab.LineAvgSpeedCheck.UpdateLineStatus+=(s,e)=>{};
  Console.WriteLine("[headless] loaded seconds="+elapsed.Elapsed.TotalSeconds.ToString("F2")+" rails="+manager.DicRailLine.Count+" oht="+manager.LstOHTNode.Count);
  if(options.ContainsKey("--load-only"))return 0;
  Dispatcher.Instance.InitLibrary();AMHS.OSS.Logic.AlgorithmMng.Instance.initSource();
  fab.PinokioLogicType=DefineEnum.LogicType.AI2_0;
  manager.UseRerouting=true;fab.RerouteNode.OnEvent();PathFinder.Instance.IsUsingRouteSelection=false;
  manager.IsUseAccelearionVer=true;manager.IsUseAlgorithmDll=false;manager.IsUseAlgorithmDllOHTAssign=false;manager.IsUseAlgorithmDllReroute=false;
  manager.IsGetTransferOHTRouteInfo=false;manager.IsGetRerouteOHTRouteInfo=false;manager.IsAssignOHTFromPython(false);manager.IsUseSendSimInfo=false;
  engine.StartDateTime=fab.StartDateTime;
  double duration=options.ContainsKey("--end-time")?double.Parse(options["--end-time"],System.Globalization.CultureInfo.InvariantCulture):(fab.EndDateTime-fab.StartDateTime).TotalSeconds;
  if(duration<=0)throw new ArgumentException("end-time must be positive");
  engine.EndDateTime=engine.StartDateTime.AddSeconds(duration);engine.InitializeEngine();
  bool python=!options.ContainsKey("--no-python");
  if(python){
   manager.IsUsePython_RouteRailCost=true;
   int port=options.ContainsKey("--port")?int.Parse(options["--port"]):9100;
   var ready=new ManualResetEvent(false);
   PServer_Python.Instance.BBiRunEnable+=(sender,e)=>{if(sender is bool&&(bool)sender)ready.Set();};
   Console.WriteLine("[headless] connecting Python 127.0.0.1:"+port);
   PServer_Python.Instance.StartServer(manager,"127.0.0.1",port,SimModelDBManager.Instance,0,(int)duration);
   if(!ready.WaitOne(TimeSpan.FromSeconds(120)))throw new TimeoutException("Python initialization timed out");
   PServer_Python.Instance.ReLoadNewModel(manager,SimModelDBManager.Instance,0,(int)duration);
   manager.FuncTCPIP_DeepLearningEndOfFrame(2);
   manager.FuncTCPIP_DeepLearningType2_NewModel(Path.GetFileName(input));
   duration=(engine.EndDateTime-engine.StartDateTime).TotalSeconds;
   Console.WriteLine("[headless] Python ready; negotiated end-time="+duration);
  }
  Console.WriteLine("[headless] effective simulation start="+engine.StartDateTime.ToString("yyyy-MM-dd HH:mm:ss")+" end="+engine.EndDateTime.ToString("yyyy-MM-dd HH:mm:ss")+" duration_s="+duration);
  SimResultDBManager.Instance.CreateSimResults(result);
  foreach(Fab f in manager.Fabs.Values){manager.UpdateRailLineCost(f);manager.UpdateNetworkLineCost2(f);manager.UpdateRouteState(f);}
  manager.SetBumpingBayAndPort();
  manager.simulatorAction=DefineEnum.SimulatorAction.RUN;
  engine.ReadyEngine(manager.SimNodes.Values.ToList(),duration);engine.RunEngine();manager.SettingEndNode.SetEvent(duration);
  long count=0;var timer=Stopwatch.StartNew();double nextProgress=0;
  Console.WriteLine("[headless] simulation started");
  while(engine.EngineState==ENGINE_STATE.RUNNING){
   // The GUI Python batch waits for SettingEndNode; ordinary native runs
   // additionally stop after the first event at the requested end time.
   if(!python&&engine.TimeNow==engine.SimEndTime){engine.EngineState=ENGINE_STATE.END;break;}
   engine.RunEvent();count++;
   if(timer.Elapsed.TotalSeconds>=nextProgress){Console.WriteLine("[headless] sim_time="+engine.TimeNow+" events="+count+" wall_s="+timer.Elapsed.TotalSeconds.ToString("F1"));nextProgress+=10;}
  }
  if(engine.EngineState!=ENGINE_STATE.END || engine.TimeNow.TotalSeconds>duration ||
     (!python && engine.TimeNow.TotalSeconds<duration))
   throw new InvalidOperationException("Engine stopped before completing its episode: "+engine.EngineState+" time="+engine.TimeNow);
  // Flush pending completed-command rows and the same final aggregate table
  // that the GUI's Run_Cycle writes before advancing to the next episode.
  SimResultDBManager.Instance.UploadCompletedCommandTrendLogs();
  if(Scheduler.Instance.CompletedCommandList.Count>0)SimResultDBManager.Instance.UploadAvgCommandTable();
  if(python)manager.FuncTCPIP_DeepLearningEndOfFrame(1);
  Console.WriteLine("[headless] completed sim_time="+engine.TimeNow+" events="+count+" wall_s="+timer.Elapsed.TotalSeconds.ToString("F2")+" result="+result);
  // Existing TCP client has an endless supervisory thread; this process owns only this episode.
  Environment.Exit(0);
  return 0;
 }
}
