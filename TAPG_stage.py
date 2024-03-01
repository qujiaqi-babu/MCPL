import argparse
import json
import os
# 使用第x张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import random
from src.data.collation import Collator
from src.data.dataset import MVSA_Dataset, Twitter_Dataset, Twitter_Dataset_no_label, MVSA_Dataset_no_label
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.generater import SequenceGeneratorModel
import src.eval_utils as eval_utils
from src.model.metrics import AESCSpanMetric, OESpanMetric

from src.model.model import MultiModalBartModelForPretrain
from src.training import pretrain
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
import torch.backends.cudnn as cudnn
DATASET_NAMES = ('MVSA', )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        default='predict_on_mvsa_by17',  # predict_on_twitter_15
                        type=str,
                        help='path to output log files, not output to file if not specified')
    parser.add_argument('--dataset',
                        default='./src/data/jsons/MVSA_no_label_info.json',
                        # default='./src/data/jsons/twitter15_info.json',
                        type=str,
                        help='json path to info of twitter dataset')
    parser.add_argument('--bart_model',
                        default='facebook/bart-base',
                        type=str,
                        help='bart pretrain model')
    parser.add_argument('--model_config',
                        default='./config/pretrain_base.json',
                        type=str,
                        help='path to load model config')
    parser.add_argument('--max_len', type=int, default=30, help='max_len')
    parser.add_argument('--max_len_a', type=float, default=0.6, help='max_len_a')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    args = parser.parse_args()
    
    # AESC
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='task type')  # 'twitter_sc' 'twitter_ae'
    parser.add_argument('--checkpoint',
                        default='refinetune_for_predict_17/2023-07-25-14-55-50-twitter_lr_3e-05/AESC',
                        type=str,
                        help='path to load AESC weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_AESC = parser.parse_args()
    
    # AE
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='twitter_ae', help='task')
    parser.add_argument('--checkpoint',
                        default='refinetune_for_predict_17/2023-07-25-14-55-50-twitter_lr_3e-05/AE',
                        type=str,
                        help='path to load AE weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_AE = parser.parse_args()
    
    # SC
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='twitter_sc', help='task')
    parser.add_argument('--checkpoint',
                        default='refinetune_for_predict_17/2023-07-25-14-55-50-twitter_lr_3e-05/SC',
                        type=str,
                        help='path to load SC weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_SC = parser.parse_args()

    return args, args_AESC, args_AE, args_SC

args, args_AESC, args_AE, args_SC = parse_args()
# device = torch.device("cuda:{}".format(1))
device = torch.device("cuda")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = os.path.join(args.log_dir, timestamp)
if args.log_dir is not None:
    os.makedirs(log_dir)
    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=1)
tokenizer = ConditionTokenizer(args)
label_ids = list(tokenizer.mapping2id.values())
senti_ids = list(tokenizer.senti2id.values())

bart_config = MultiModalBartConfig.from_dict(
    json.load(open(args.model_config)))

# 特殊符号
bos_token_id = 0
eos_token_id = 1

# AESC
seq2seq_model = MultiModalBartModel_AESC.from_pretrained(args_AESC.checkpoint, 
                                                        config=bart_config,
                                                        args=args_AESC,
                                                        bart_model=args.bart_model,
                                                        tokenizer=tokenizer,
                                                        label_ids=label_ids)
model_AESC = SequenceGeneratorModel(seq2seq_model,
                               bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=args.max_len,
                               max_len_a=args.max_len_a,
                               num_beams=args.num_beams,
                               do_sample=False,
                               repetition_penalty=1,
                               length_penalty=1.0,
                               pad_token_id=eos_token_id,
                               restricter=None)
model_AESC.to(device)
model_AESC.eval()

# AE
seq2seq_model = MultiModalBartModel_AESC.from_pretrained(args_AE.checkpoint, 
                                                        config=bart_config,
                                                        args=args_AE,
                                                        bart_model=args.bart_model,
                                                        tokenizer=tokenizer,
                                                        label_ids=label_ids)
model_AE = SequenceGeneratorModel(seq2seq_model,
                                bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=args.max_len,
                                max_len_a=args.max_len_a,
                                num_beams=args.num_beams,
                                do_sample=False,
                                repetition_penalty=1,
                                length_penalty=1.0,
                                pad_token_id=eos_token_id,
                                restricter=None)
model_AE.to(device)
model_AE.eval()

# SC
seq2seq_model = MultiModalBartModel_AESC.from_pretrained(args_SC.checkpoint, 
                                                        config=bart_config,
                                                        args=args_SC,
                                                        bart_model=args.bart_model,
                                                        tokenizer=tokenizer,
                                                        label_ids=label_ids)
model_SC = SequenceGeneratorModel(seq2seq_model,
                                bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=args.max_len,
                                max_len_a=args.max_len_a,
                                num_beams=args.num_beams,
                                do_sample=False,
                                sc_only=True,
                                repetition_penalty=1,
                                length_penalty=1.0,
                                pad_token_id=eos_token_id,
                                restricter=None)
model_SC.to(device)
model_SC.eval()

logger.info('Public', pad=True)
for k, v in vars(args).items():
    logger.info('{}: {}'.format(k, v))
logger.info('AESC', pad=True)
for k, v in vars(args_AESC).items():
    logger.info('{}: {}'.format(k, v))
logger.info('AE', pad=True)
for k, v in vars(args_AE).items():
    logger.info('{}: {}'.format(k, v))
logger.info('SC', pad=True)
for k, v in vars(args_SC).items():
    logger.info('{}: {}'.format(k, v))

# dataset = Twitter_Dataset_no_label(args.dataset, split='train')
dataset = MVSA_Dataset_no_label(args.dataset)
targetid2mapping = {value:key for key, value in tokenizer.mapping2targetid.items()}

json_list = []
target_shift = len(tokenizer.mapping2targetid) + 2
cnt_generate = 0  # 记录生成的个数
cnt_first_save = 0  # 记录第一次过滤后的个数
cnt_second_save = 0  # 记录第二次过滤后的个数
cnt_final_save = 0  # 记录最终保存的个数
for sample in tqdm(dataset):
    if sample['image_id'] != "3590.jpg":
        continue
    logger.info('image_id:{}'.format(sample['image_id']), pad=True)
    logger.info('sentence:{}'.format(sample['sentence']))
    batch = [sample]

    image_features = [
        torch.from_numpy(x['img_feat'][:49])
        if 'img_feat' in x else torch.empty(0) for x in batch
    ]  # 有的输入样本可能不含图片，image_feature置为空tensor
    
    img_num = [36 for x in image_features]  # len(x)
    # print(img_num)
    # exit()

    target = [x['sentence'] for x in batch]
    sentence = list(target)

    encoded_conditions = tokenizer.encode_condition(
        img_num=img_num, sentence=sentence, text_only=False)

    input_ids = encoded_conditions['input_ids']  # token to idex
    condition_img_mask = encoded_conditions['img_mask']    
    
    output = {}
    output['attention_mask'] = encoded_conditions['attention_mask']
    output['image_features'] = image_features
    output['input_ids'] = input_ids
    aesc_infos = {'labels':torch.tensor([[0, tokenizer.mapping2targetid['AESC'], tokenizer.mapping2targetid['AESC']]])}
    ae_infos = {'labels':torch.tensor([[0, tokenizer.mapping2targetid['AE'], tokenizer.mapping2targetid['AE']]])}
    sc_infos = {'labels':torch.tensor([[0, tokenizer.mapping2targetid['SC'], tokenizer.mapping2targetid['SC']]])}
    # generate target-sentiment span
    predict = model_AESC.predict(
        input_ids=output['input_ids'].to(device),
        image_features=list(
            map(lambda x: x.to(device), output['image_features'])),
        attention_mask=output['attention_mask'].to(device),
        aesc_infos=aesc_infos)[0]  # 因为只含1个样本
    logger.info('AESC:{}'.format(predict))

    target_list_AESC = []
    target_emotion_list_AESC = []
    span_num = int((len(predict) - 3 - 1) / 3)  # target-sentiment对的个数
    cnt_generate += span_num  # 记录生成的个数
    for i in range(span_num):
        target_list_AESC.append(tuple(predict[3+3*i:5+3*i].cpu().numpy()))
        target_emotion_list_AESC.append(tuple(predict[3+3*i:6+3*i].cpu().numpy()))

    # generate target span
    predict = model_AE.predict(
        input_ids=output['input_ids'].to(device),
        image_features=list(
            map(lambda x: x.to(device), output['image_features'])),
        attention_mask=output['attention_mask'].to(device),
        aesc_infos=ae_infos)[0]  # 因为只含1个样本
    logger.info('AE:{}'.format(predict))
    
    target_list_AE = []
    span_num = int((len(predict) - 3 - 1) / 2)  # target-sentiment对的个数
    for i in range(span_num):
        target_list_AE.append(tuple(predict[3+2*i:5+2*i].cpu().numpy()))

    # 第一次过滤
    first_filter_result = []
    for target_AESC, target_emotion_AESC in zip(target_list_AESC, target_emotion_list_AESC):
        if target_AESC in target_list_AE:
            cnt_first_save += 1  # 记录第一次过滤后的个数
            first_filter_result.append(target_emotion_AESC)
            sc_infos['labels'] = torch.cat([sc_infos['labels'], torch.tensor([target_emotion_AESC])], dim=1)
    
    # generate given target sentiment span
    sc_infos['labels'] = torch.cat([sc_infos['labels'], torch.tensor([[1]])], dim=1)
    predict = model_SC.predict(
        input_ids=output['input_ids'].to(device),
        image_features=list(
            map(lambda x: x.to(device), output['image_features'])),
        attention_mask=output['attention_mask'].to(device),
        aesc_infos=sc_infos)[0]  # 因为只含1个样本
    logger.info('SC:{}'.format(predict))

    target_emotion_list_SC = []
    span_num = int((len(predict) - 3 - 1) / 3)  # target-sentiment对的个数
    for i in range(span_num):
        target_emotion_list_SC.append(tuple(predict[3+3*i:6+3*i].cpu().numpy()))

    # 第二次过滤
    second_filter_result = []
    for target_emotion_AESC in first_filter_result:
        if target_emotion_AESC in target_emotion_list_SC:
            cnt_second_save += 1  # 记录第二次过滤后的个数
            second_filter_result.append(target_emotion_AESC)

    # 生成json文件
    sentence_split = sample['sentence'].split()
    sample_dict = {}
    sample_dict["words"] = sentence_split
    sample_dict["image_id"] = sample['image_id']
    sample_dict["aspects"] = []

    word_bpes = [[tokenizer.begin_text_id]]
    for word in sentence_split:
        bpes = tokenizer._base_tokenizer.tokenize(word,
                                                  add_prefix_space=True)
        bpes = tokenizer.convert_tokens_to_ids(bpes)
        word_bpes.append(bpes)
    word_bpes.append([tokenizer.end_text_id])
    lens = list(map(len, word_bpes))
    cum_lens = np.cumsum(list(lens)).tolist()  # 第i个word从input_idx的cum_lens[i]开始
    flag = 0  # 记录是否有生成合法标签
    for s_bpe, e_bpe, polarity in second_filter_result:  # (28, 30,  5)
        for i in range(len(cum_lens)):
            if cum_lens[i] == s_bpe - target_shift:
                for j in range(i+1, len(cum_lens)):
                    if cum_lens[j-1] == e_bpe - target_shift:
                        cnt_final_save += 1  # 记录最终保存的个数
                        flag = 1
                        aspect = {}
                        aspect["from"] = i
                        aspect["to"] = j
                        aspect["polarity"] = targetid2mapping[polarity]
                        aspect["term"] = sentence_split[i:j]
                        sample_dict["aspects"].append(aspect)
                        break
                break
    if flag:
        json_list.append(sample_dict)
    # break  # 只生成第一个样本的结果
json_str = json.dumps(json_list)
with open(os.path.join(log_dir, 'result.json'), 'w') as json_file:
	json_file.write(json_str)
logger.info('cnt_generate:{}'.format(cnt_generate))
logger.info('cnt_first_save:{}'.format(cnt_first_save))
logger.info('cnt_second_save:{}'.format(cnt_second_save))
logger.info('cnt_final_save:{}'.format(cnt_final_save))
