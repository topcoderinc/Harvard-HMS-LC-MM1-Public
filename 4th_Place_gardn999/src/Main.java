
public class Main{
  
  public static void main(String[] args){
    String sourceDir, 
           solutionPath = "solution.csv";
    
    if (args.length < 1 || args.length > 2){
      System.out.println("HMS Lung Cancer - LungTumorTracer");
      System.out.println("usage: java -jar LungTumorTracer.jar "
           + "<sourceDir> <solutionPath>");
      System.out.println("<sourceDir> - directory containing the CT scans "
           + "with images in PNG format");
      System.out.println("<solutionPath> - optional name of output "
           + "file  (\"solution.csv\" by default)");
    }else{ 
      sourceDir = args[0];    
      if (args.length > 1) solutionPath = args[1];
      char end = sourceDir.charAt(sourceDir.length()-1);
      if (end != '/' && end != '\\') sourceDir = sourceDir + "/";
      
      TumorFinder tf = new TumorFinder();
      tf.run(sourceDir, solutionPath);
    }
  }

}
