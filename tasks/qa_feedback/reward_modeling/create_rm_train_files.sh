####对原始qa文件做sequence、token粒度拆解，并对token粒度做label标记
#输入文件/home/chenzhengzong/FineGrainedRLHF/tasks/qa_feedback/data/train_feedback.json
#关注1、question；2、"prediction 1"、3、feedback->errors(span为字符串索引粒度)

# 脚本中以空格为分割，来划分token，改为中文时，需要做如下操作：
# 1、from transformers import AutoTokenizer,LlamaTokenizer
# 2、tokenizer=LlamaTokenizer.from_pretrained('/home/chenzhengzong/from_nlp_group/for_sft_infer/live_script/llama-13b-gpt4_llm_comparision2_clean-154855')
# 3、a="\'首先，我们来看看这款冷吃脆三拼的包装。它采用了红色的礼盒设计，非常喜庆。无论是自己品尝还是送给亲朋好友，都非常合适。而且，它的包装也非常有特色，让人一眼就能看出是地道的四川美食。\n接下来，让我们来聊聊这款冷吃脆三拼的味道。它采用了传统的四川制作方法，每一口都充满了浓浓的川渝味。毛肚的爽脆，鸭肠的鲜嫩，贡菜的清甜，三种不同的口味，三种不同的口感，让你一次性满足对冷吃的一切幻想。而且，它的辣度也刚刚好，不会过于刺激，非常适合爱吃辣又怕辣的朋友。\n最后，我们来看看这款冷吃脆三拼的价格。原价59.9的，现在厂家周年庆，直接降价20，只要39.9就能买到这么一大盒，性价比超高！而且，它的份量也非常足，足够一家人分享。\n总的来说，这款冷吃脆三拼是一款非常值得购买的美食。无论是从包装、味道还是价格来看，它都是你的不二之选。如果你也是冷吃的爱好者，那么赶紧下单吧，让这份美味陪伴你度过美好的时光。\'"
# 4、c=tokenizer(a)
# 5、tokenizer.convert_ids_to_tokens(c['input_ids'])

set -e

python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level subsentence --error_category NF-ERR --ignore_context --data_dir ./tasks/qa_feedback/data/
python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level sentence --error_category F-ERR --data_dir ./tasks/qa_feedback/data/
python tasks/qa_feedback/reward_modeling/create_comp_rm_files.py --input_dir ./tasks/qa_feedback/data/ --output_dir ./tasks/qa_feedback/data/COMP_sequence/