from torch import cuda
from numba import cuda as cuda_n

cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128
    }

if (cuda.is_available()):
  print("CUDA is available")
  device = cuda_n.get_current_device()
  my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
  my_cc = device.compute_capability
  cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
  total_cores = cores_per_sm*my_sms
  print("GPU compute capability: " , my_cc)
  print("GPU total number of  Streaming MultiProcessors (SMs): " , my_sms)
  print("Total Stream Processor (SP) cores: " , total_cores)
else:
  print("CUDA is NOT available")

# 4070:
#   GPU compute capability:  (8, 9)
#   GPU total number of  Streaming MultiProcessors (SMs):  46
#   Total Stream Processor (SP) cores:  5888