# 最好用绝对路径
rm -r /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin
mkdir /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin
cd /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/src
nvcc -O3 -std=c++14 repeater.cu -arch compute_86 -Xcompiler -fopenmp -o /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin/output
echo -e “Compilation successful, starting execution”
/home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin/output