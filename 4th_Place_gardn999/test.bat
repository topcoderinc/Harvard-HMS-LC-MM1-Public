
@echo off

REM requires Java version 1.7 or later

echo running LungTumorTracer.jar

REM parameters: 
REM %1 - <sourceDir> - directory containing the CT scans with images in PNG format
REM %2 - <solutionPath> - optional name of output file  ("solution.csv" by default)

java -jar LungTumorTracer.jar %1 %2