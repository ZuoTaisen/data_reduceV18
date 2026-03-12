#!/bin/bash
beginTime=`date +%s`
##########################################################在这里输入初始值 
startNum2=20825   #在这里修改起始run号
#########################################################
#startNum2Plus=(0 7 11)   #选择要运行的runk与，startNum2+
#!/bin/bash

# 定义一个数组来存储文件中的数据
declare -a data_array

# 读取文件中的数据，并将其逐行存入数组
index=0
while IFS= read -r line; do
    data_array[$index]="$line"
    ((index++))
done < "RunNums.txt"

# 循环打印数组中的每个元素
#for item in "${data_array[@]}"; do
#    echo "$item"
#done

for item in "${data_array[@]}"
#for i in "${startNum2Plus[@]}"
######################################################
#num=2
#for i in `seq 0 $num`
########################################################
do
        {
                runNum=$item #`expr \( ${i} + ${startNum2} \)`
                len=${#runNum}
                zeros_needed=$((7 - len))
                [ $zeros_needed -lt 0 ] && zeros_needed=0
                 printf -v zeros '%0*d' $zeros_needed 0
                startRun2="RUN${zeros}${runNum}"

                #startRun2="RUN$(printf '%07d'$RunNum)" #"RUN$(printf '%0*.*d'$((7-${#runNum})) 0)$runNum"
                #startRun="RUN00"${runNum}
                echo $i  "Start the ${startRun2} run and the time is:" `date "+%Y-%m-%d %H:%M:%S"`

#####################################################################################在这里输入在画的内容

                python plot_data_D3.py ${startRun2} 2D     
                #python plot_data_D2.py ${startRun} 2D
                #python plot_data_D1.py ${startRun} 2D
# 在这里修改要画出来的内容，lambdar 位置的可选择项包括：lambda,tof,2D,2DQ,xy; 
# 还可以是某一个Bank的波长或者二维谱，如：D11 lambda, D22 2D; 2DQ指的是QX，QY的二维图。


######################################################################################

                startNum2=`expr $i + ${startNum}`            

                echo $i  "Complete the ${startRun2} run and the time is :" `date "+%Y-%m-%d %H:%M:%S"`
                echo "-----------------------------------------------------------"
        # 结尾的&确保每个进程后台执行
        }&
done
endTime=`date +%s`
echo "Total time consumed:" $(($endTime-$beginTime)) "Seconds"

