cd ~/Documents/Code/KinectFusion/build
cmake ..
make 2>log 1>log
cd ../bin
sudo ./testdisplay 1>out
