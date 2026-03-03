path=$1

echo 'Number of finished analyses in S3_truncated'$1
find S3_truncated$1 -name '*inference.txt' -print | wc -l
echo 'Number of fit folders in Condition 100:'
find S3_truncated$1 -type d -path '*/100/fit'$1 -print | wc -l
echo 'Number of fit folders in Condition 500:'
find S3_truncated$1 -type d -path '*/500/fit'$1 -print | wc -l

