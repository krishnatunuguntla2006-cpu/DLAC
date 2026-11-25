
=== docs/sample_trace_cpu.json ===
Source : MEASURED: instructor's Session 3 run, CPU only, 512 x 512 PNG
Device : cpu | runs used: 1

 #  stage               category                 ms   share
 1  receive upload      Serving / I/O          0.12    0.0%  
 2  decode image        CPU preprocessing     36.67    8.4%  ###
 3  letterbox resize    CPU preprocessing      4.17    1.0%  
 4  to tensor           CPU preprocessing      7.50    1.7%  #
 5  host to device      Memory transfer        0.05    0.0%  
 6  model forward       Compute              315.15   71.9%  #############################
 7  NMS                 Postprocessing        70.59   16.1%  ######
 8  rescale boxes       Postprocessing         2.81    0.6%  
 9  serialise response  Serving / I/O          1.52    0.3%  

Sum of stage times: 438.58 ms (recorded total 438.76 ms)

By category:
  Serving / I/O          1.64 ms    0.4%  
  CPU preprocessing     48.34 ms   11.0%  ####
  Memory transfer        0.05 ms    0.0%  
  Compute              315.15 ms   71.9%  #############################
  Postprocessing        73.40 ms   16.7%  #######

Largest stage: model forward (Compute), 315.15 ms = 71.9% of the measured time
PROFILE FIRST: model forward. Any saving in the largest stage has the largest possible effect.

Rules of thumb applied to this trace:
  - This ran on the CPU, so 'host to device' costs almost nothing. On a GPU machine it will not be zero, and the compute stage should shrink.
  - Compute is over half of the time. Faster precision (FP16, INT8), TensorRT and a GPU are the levers (Sessions 13 and 14).
  - Postprocessing is over 15% of the time. NMS cost grows with the number of candidate boxes.
  - This is a single run, so it includes warm-up. Use collect_trace.py to measure repeated runs.
  - Not measured here: time waiting in a queue and network time. They appear when a server such as Triton is added.

What if 'model forward' were faster? (other stages unchanged)
   2x faster ->  281.00 ms total, overall speed-up 1.56x
  10x faster ->  154.94 ms total, overall speed-up 2.83x
  free (0 ms)  ->  123.43 ms total, overall speed-up 3.55x  <- the limit

Throughput for one request at a time: 2.28 images per second

=== docs/sample_trace_gpu_example.json ===
Source : ILLUSTRATIVE: instructor example for a GPU machine, 1280 x 720 image. NOT measured.
Device : cuda | runs used: 1

 #  stage               category                 ms   share
 1  receive upload      Serving / I/O          0.20    0.8%  
 2  decode image        CPU preprocessing      6.50   25.5%  ##########
 3  letterbox resize    CPU preprocessing      2.10    8.2%  ###
 4  to tensor           CPU preprocessing      1.40    5.5%  ##
 5  host to device      Memory transfer        0.90    3.5%  #
 6  model forward       Compute               11.80   46.3%  ###################
 7  NMS                 Postprocessing         2.20    8.6%  ###
 8  rescale boxes       Postprocessing         0.30    1.2%  
 9  serialise response  Serving / I/O          0.10    0.4%  

Sum of stage times: 25.50 ms (recorded total 25.60 ms)

By category:
  Serving / I/O          0.30 ms    1.2%  
  CPU preprocessing     10.00 ms   39.2%  ################
  Memory transfer        0.90 ms    3.5%  #
  Compute               11.80 ms   46.3%  ###################
  Postprocessing         2.50 ms    9.8%  ####

Largest stage: model forward (Compute), 11.80 ms = 46.3% of the measured time
PROFILE FIRST: model forward. Any saving in the largest stage has the largest possible effect.

Rules of thumb applied to this trace:
  - CPU preprocessing is over 25% of the time. A fast GPU would sit waiting for it. Look at decoding, resizing and tensor conversion before tuning the model.
  - This is a single run, so it includes warm-up. Use collect_trace.py to measure repeated runs.
  - Not measured here: time waiting in a queue and network time. They appear when a server such as Triton is added.

What if 'model forward' were faster? (other stages unchanged)
   2x faster ->   19.60 ms total, overall speed-up 1.30x
  10x faster ->   14.88 ms total, overall speed-up 1.71x
  free (0 ms)  ->   13.70 ms total, overall speed-up 1.86x  <- the limit

Throughput for one request at a time: 39.22 images per second

=== COMPARISON: share of time by category ===
category               sample_trace_cpu  sample_trace_gpu_e
Serving / I/O                      0.4%                1.2%
CPU preprocessing                 11.0%               39.2%
Memory transfer                    0.0%                3.5%
Compute                           71.9%               46.3%
Postprocessing                    16.7%                9.8%
total (ms)                       438.58               25.50

The bottleneck moves when the hardware changes. Optimise the stage that is largest on the machine you will actually serve from.
