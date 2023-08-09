# 最好用绝对路径
rm -r /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/build
mkdir /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/build
rm -r /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin
mkdir /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin
cd /home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/build
cmake ..
make 
/home/ivyadmin/Project_demo/repeter_Means/CUDA_Test/bin/Repeater