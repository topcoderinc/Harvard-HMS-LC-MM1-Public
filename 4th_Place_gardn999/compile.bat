
@echo off

REM requires Java version 1.7 or later

if not exist build mkdir build

javac src\*.java -d build

chdir build

jar cfe ..\LungTumorTracer.jar Main *.class

chdir ..

echo LungTumorTracer.jar successfully created.
