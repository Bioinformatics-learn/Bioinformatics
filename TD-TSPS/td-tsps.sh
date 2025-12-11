# -*-coding:gb2312-*-

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

echo "************************  Begin  ************************"

python TD-TSPS.py  chr21.fa  testsorted.bam  testdiscordants_sorted.bam

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s - $startTime_s ]

echo "$startTime ---> $endTime" "Total:$sumTime seconds"
echo $sumTime >> time.ods

echo "************************  Finish  ************************"
