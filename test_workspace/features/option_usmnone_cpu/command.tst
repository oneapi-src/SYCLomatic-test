================= libcu_atomic ==================
dpct --cuda-include-path=/usr/local/cuda-11.4/include /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/libcu_atomic.cu --in-root /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic --out-root /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root --usm-level=none
dpcpp  -c /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root/libcu_atomic.dp.cpp 
sycl-ls
================= libcu_atomic ==================
dpct --cuda-include-path=/usr/local/cuda-11.4/include /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/libcu_atomic.cu /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root/libcu_atomic.dp.cpp --in-root /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic --out-root /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root --usm-level=none
dpcpp  -c /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root/libcu_atomic.dp.cpp 
dpcpp  -c /mnt/hdd/wenhuini/workstations/SYCLomatic-test/test_workspace/features/option_usmnone_cpu/libcu_atomic/out_root/out_root/libcu_atomic.dp.cpp 
sycl-ls
