// Patch only a private copy of the installed, hash-pinned transport assembly.
// No simulation event, routing, message layout, or license code is changed.
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
class PatchTransport {
 static void Main(string[] args) {
  using(var hash=SHA256.Create())using(var stream=File.OpenRead(args[0])) {
   var actual=BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "");
   if(actual!="DD470F9C42DDA3D9EA4838D665D86DC64E6A5E2B8084F4E4030A17761CFC8DA3")
    throw new Exception("Installed transport version changed; inspect it before adapting this patch. SHA256="+actual);
  }
  using(var module=ModuleDefinition.ReadModule(args[0])) {
   var type=module.Types.Single(t=>t.FullName=="Pinokio.TCP.IP.PServer_Python");
   var method=type.Methods.Single(m=>m.Name=="StartingServerLocal");
   var jumps=method.Body.Instructions.Where(i=>i.Operand is Instruction && ((Instruction)i.Operand).Offset<i.Offset);
   var target=jumps.Select(i=>(Instruction)i.Operand).OrderBy(i=>i.Offset).First();
   if(target.OpCode!=OpCodes.Nop||target.Offset>16)throw new Exception("Unexpected transport loop layout");
   var il=method.Body.GetILProcessor();
   var delay=il.Create(OpCodes.Ldc_I4,5);
   var sleep=il.Create(OpCodes.Call,module.ImportReference(typeof(System.Threading.Thread).GetMethod("Sleep",new[]{typeof(int)})));
   il.InsertAfter(target,delay);il.InsertAfter(delay,sleep);
   module.Write(args[1]);
   Console.WriteLine("Private TCP transport: 5 ms wait in connection supervisor (no per-tick delay).");
  }
 }
}
