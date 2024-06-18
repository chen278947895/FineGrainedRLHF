####对原始qa文件做sequence、token粒度拆解，并对token粒度做label标记
#输入文件/home/chenzhengzong/FineGrainedRLHF/tasks/qa_feedback/data/train_feedback.json
#关注1、question；2、"prediction 1"、3、feedback->errors(span为字符串索引粒度)
set -e

python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level subsentence --error_category NF-ERR --ignore_context --data_dir ./tasks/qa_feedback/data/
python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level sentence --error_category F-ERR --data_dir ./tasks/qa_feedback/data/
python tasks/qa_feedback/reward_modeling/create_comp_rm_files.py --input_dir ./tasks/qa_feedback/data/ --output_dir ./tasks/qa_feedback/data/COMP_sequence/