#!/bin/sh
usage(){
      echo "
      Before running this script, please modify the parameters in this script.
      
      -s is start index of CT image, please input number, <= 0 will running in all data
      -e is end index of CT image, please input number
      -h is help
      
      example 1: sh ./run_matlab_brain_extract.sh -s 1 -e 20
      example 2: sh ./run_matlab_brain_extract.sh -s -1
      
      The example 1 will applying brain extracction on 1~20 CT image in source dir
      The example 2 will applying brain extracction on all CT image in source dir
       "
}

main(){
while getopts "s:e:h" arg; do
  case $arg in
    s)
      start_index=$OPTARG;;
    e)
      end_index=$OPTARG;;
    h)

      exit 0
      ;;
    ?)
      echo "Unsupport input parameter!"
      exit 1
      ;;
  esac
done

# check input args
if test -z "$start_index";then
	echo "start_index is null string or start_index not exist"
  echo "using all CT images in source dir!"
  start_index=-1
  end_index=-1
fi

# Please using absolute path
# set parameter of this script -------------> Modify by user
matlab_path=/gpfs/lab/liangmeng/members/liyifan/matlab2016b/bin/matlab  # your matlab path
batch_script_name=SkullStripSingleCT_MF_batch
log_dir=/gpfs/lab/liangmeng/members/liyifan/git/python_programm/REUnet/example_data/log

# set parameter in matlab -------------> Modify by user
StripSkullCT_code_dir=/gpfs/lab/liangmeng/members/liyifan/matlab_toolbox/StripSkullCT-master  # StripSkullCT-master path
source_dir=/gpfs/lab/liangmeng/members/liyifan/git/python_programm/REUnet/example_data/image
out_dir=/gpfs/lab/liangmeng/members/liyifan/git/python_programm/REUnet/example_data/image_be

# mkdir out dir ----------------------------------------------------------------------------------------------------------
if [ ! -d "${log_dir}" ];then
  mkdir -p ${log_dir}  # mkdir multi level dir by arg -p
  echo "mkdir: ${log_dir}"
fi

# run batch BE
${matlab_path} -nodesktop -nosplash -r "StripSkullCT_code_dir='${StripSkullCT_code_dir}'; source_dir='${source_dir}'; out_dir='${out_dir}'; start_index=${start_index}; end_index=${end_index}; ${batch_script_name}; quit;" > ${log_dir}/run_log_${start_index}_${end_index}_`date +%Y%m%d-%H%M%S`.txt < /dev/null
}

main $@
