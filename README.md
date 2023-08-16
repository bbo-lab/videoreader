# videoreader
Videoreader, currently a pyav v3 wrapper, with the following features:

 - Reading CCV-Files
 - Preemptive cached video access
 - Simoultanous threaded reading of multiple video files
 - Effects applied by a pipelined graph-structure

### Syntax
The syntax is mostly orientated on ffmpeg, generally [\<input1\>][\<input2\>]\<effect=arg0=\<arg0\>\>[\<output\>] 
 - Effects are sepperated by simikolon
 - Inputs/Outputs of effects are defined in squared brackets
 - arguments can be defined by effect

### run_pipeline.py
This is a simple tool if you want to use the videoreader from the commandline. You can define the following options:
 - '-i', '--input' \<Can define multiple input-files\>
 - '-o', '--output' \<Define where to write output\>
 - '-g', '--filtergraph' \<The filtegraph which is applied\>

#### Examples:
 - python3 svidreader/run_pipeline.py -i ../test/cubes.mp4 --filtergraph "contrast" --output test.csv
#### List of effects:
 - scale\
   Scales input by a given factor\
   scale: float, factor of scaling
 - arange\
   Aranges multiple inputs into a grid\
   ncols: int, number of columns
 - viewer\
   Views the last read input
   backend: string, Backend to be used, values are matplotlib, ffplay, opencv
 - bgr2gray\
   Converts each RGB-input-frame to three grayscale-outputs
 - cache\
   Adds a cache with saves the last shown frames
 - tblend\
   Subtraction of the previous to the current frame
 - perprojection\
   Perspective projection from different camera-models
