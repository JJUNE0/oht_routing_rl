using System;
using System.Linq;
using Mono.Cecil;

// Inspect metadata without loading GUI dependencies or constructing a window.
internal static class VerifyFramework {
 static string Target(string path) {
  using(var assembly=AssemblyDefinition.ReadAssembly(path)) {
   var attribute=assembly.CustomAttributes.SingleOrDefault(a=>a.AttributeType.FullName=="System.Runtime.Versioning.TargetFrameworkAttribute");
   return attribute==null?null:(string)attribute.ConstructorArguments[0].Value;
  }
 }
 static int Main(string[] args) {
  try {
   if(args.Length!=2)throw new ArgumentException("Expected original GUI and headless assembly paths");
   string gui=Target(args[0]),headless=Target(args[1]);
   if(gui!=".NETFramework,Version=v4.8"||headless!=gui)throw new InvalidOperationException("Framework target mismatch: GUI="+gui+" headless="+headless);
   Console.WriteLine("[build] GUI/headless framework targets match: "+gui);
   return 0;
  } catch(Exception e) {Console.Error.WriteLine(e.Message);return 1;}
 }
}
