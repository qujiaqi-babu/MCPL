import argparse
import json
import os
# 使用第x张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from src.data.dataset import MVSA_Dataset_with_label, Twitter_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.generater import SequenceGeneratorModel
import src.eval_utils_co_training as eval_utils  # Babu
from src.model.metrics import AESCSpanMetric, OESpanMetric

from src.model.model import MultiModalBartModelForPretrain
from src.training import co_training
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
import torch.backends.cudnn as cudnn
DATASET_NAMES = ('MVSA', )


def main(rank, args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.MVSA_enabled and args.Twitter_enabled:
        checkpoint_path = os.path.join(args.log_dir, timestamp+'-lr_{}-lamb_{}'.format(args.lr, args.lamb))
        log_dir = os.path.join(args.log_dir, timestamp+'-lr_{}-lamb_{}'.format(args.lr, args.lamb))
    # elif args.MVSA_enabled:
    #     checkpoint_path = os.path.join(args.log_dir, timestamp+'-lamb_{}'.format(args.lamb))
    #     log_dir = os.path.join(args.log_dir, timestamp+'-lamb_{}'.format(args.lamb))
    elif args.Twitter_enabled:
        checkpoint_path = os.path.join(args.log_dir, timestamp+'-lr_{}'.format(args.lr))
        log_dir = os.path.join(args.log_dir, timestamp+'-lr_{}'.format(args.lr))
    else:
        raise ValueError('at least one dataset should be enabled')
    tb_writer = None
    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=os.path.join('runs', log_dir))  # Babu

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=(rank == 0))

    # make checkpoint dir if not exist
    if rank == 0 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    # logger.info('Public', pad=True)

    if args.lamb == 0:
        args.MVSA_enabled = 0

    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    # logger.info('AESC', pad=True)
    # for k, v in vars(args_AESC).items():
    #     logger.info('{}: {}'.format(k, v))
    # logger.info('AE', pad=True)
    # for k, v in vars(args_AE).items():
    #     logger.info('{}: {}'.format(k, v))
    # logger.info('SC', pad=True)
    # for k, v in vars(args_SC).items():
    #     logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        # device = torch.device("cuda:{}".format(0))
        device = torch.device("cuda")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer(args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())

    bart_config = MultiModalBartConfig.from_dict(
        json.load(open(args.model_config)))
    
    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout
    
    # 特殊符号
    bos_token_id = 0
    eos_token_id = 1

    if args.checkpoint:
        pretrain_model = MultiModalBartModelForPretrain.from_pretrained(
            args.checkpoint,
            config=bart_config,
            bart_model=args.bart_model,
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False)
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        seq2seq_model.encoder.load_state_dict(
            pretrain_model.encoder.state_dict())
        seq2seq_model.decoder.load_state_dict(
            pretrain_model.span_decoder.state_dict())
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,  # 生成句子的最大长度
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,  # beam search的大小
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
    else:
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        model = SequenceGeneratorModel(seq2seq_model,
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

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    scaler = GradScaler() if args.amp else None

    epoch = 0
    # =========================== data =============================

    logger.info('Loading data...')

    collate_aesc = Collator(tokenizer,
                            mlm_enabled=False,
                            mrm_enabled=False,
                            senti_enabled=False,
                            ae_enabled=False,
                            oe_enabled=False,
                            aesc_enabled=True,
                            anp_enabled=False,
                            text_only=args.text_only)
    collate_ae = Collator(tokenizer,
                          mlm_enabled=False,
                          senti_enabled=False,
                          ae_enabled=False,
                          oe_enabled=False,
                          aesc_enabled=False,
                          anp_enabled=False,
                          twitter_ae_enabled=True,
                          text_only=args.text_only)
    collate_sc = Collator(tokenizer,
                          mlm_enabled=False,
                          senti_enabled=False,
                          ae_enabled=False,
                          oe_enabled=False,
                          aesc_enabled=False,
                          anp_enabled=False,
                          twitter_sc_enabled=True,
                          text_only=args.text_only)

    MVSA_dataset = MVSA_Dataset_with_label(args.MVSA_dataset) if args.MVSA_enabled else None
    train_dataset = Twitter_Dataset(args.dataset, split='train') if args.Twitter_enabled else None
    dev_dataset = Twitter_Dataset(args.dataset, split='dev')
    test_dataset = Twitter_Dataset(args.dataset, split='test')

    task_type = ['AESC', 'AE', 'SC']
    task_enbled = [args.aesc_enabled, args.ae_enabled, args.sc_enabled]
    collate_list = [collate_aesc, collate_ae, collate_sc]
    metric_type = []
    metric_type.append(AESCSpanMetric(eos_token_id, num_labels=len(label_ids), conflict_id=-1))
    metric_type.append(OESpanMetric(eos_token_id, num_labels=len(label_ids)))
    metric_type.append(AESCSpanMetric(eos_token_id, num_labels=len(label_ids), conflict_id=-1)) 

    # ========================== training ============================

    logger.info('Start training', pad=True)
    scaler = GradScaler() if args.amp else None

    task_list = []  # 多任务列表
    metric_list = []  # 多任务度量列表
    best_dev_res = []  # 多任务在dev集上的测试结果
    best_dev_test_res_list = []  # 在dev集上某任务最优的模型在test上的各任务测试结果
    best_dev_epoch = []  # 在dev集上最优的epoch（从1开始）
    best_dev_dataset = []  # 在dev集上最优的dataset（Twitter or MVSA）
    train_loaders_twitter = []
    train_loaders_mvsa = []
    dev_loaders_twitter = []
    test_loaders_twitter = []
    for ty, enable, collate_t, metric_t in zip(task_type, task_enbled, collate_list, metric_type):
        if enable:
            task_list.append(ty)
            metric_list.append(metric_t)
            best_dev_res.append(None)
            best_dev_test_res_list.append(None)
            best_dev_epoch.append(0)
            best_dev_dataset.append('Twitter')
            loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=collate_t) if args.Twitter_enabled else None
            train_loaders_twitter.append(loader)
            loader = DataLoader(dataset=MVSA_dataset,
                                batch_size=args.batch_size*4,  # 4倍于Twitter
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=collate_t) if args.MVSA_enabled else None
            train_loaders_mvsa.append(loader)
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    collate_fn=collate_t)
            dev_loaders_twitter.append(dev_loader)
            test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    collate_fn=collate_t)
            test_loaders_twitter.append(test_loader)
    logger.info('task_list:{}'.format(task_list)) 

    start = datetime.now()
    epoch = 0
    train_enable_l = [args.Twitter_enabled, args.MVSA_enabled]  # 是否启用该数据集作为训练集
    dataset_l = ['Twitter', 'MVSA']
    train_loaders_l = [train_loaders_twitter, train_loaders_mvsa]
    mvsa_l = [False, True]
    log_interval_l = [10, 100]
    t = 0
    while epoch < args.epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        for train_enable, dataset, train_loaders, mvsa, log_interval in zip(train_enable_l, dataset_l, train_loaders_l, mvsa_l, log_interval_l):
            if train_enable:
                t += 1
                logger.info('Train on Dataset {}'.format(dataset), pad=True)
                co_training(task_list=task_list,
                        epoch=epoch,
                        model=model,
                        train_loaders=train_loaders,
                        optimizer=optimizer,
                        device=device,
                        args=args,
                        mvsa=mvsa,  # 是否在mvsa上训练
                        logger=logger,
                        log_interval=log_interval,  # 每log_interval个step记录一下
                        tb_writer=tb_writer,
                        tb_interval=log_interval,
                        scaler=scaler)
                print('test!!!!!!!!!!!!!!')
                if (epoch + 1) % args.eval_every == 0:
                    # res_dev_list = []  # 当前模型在dev上各任务的性能
                    res_test_list = []  # 当前模型在test上各任务的性能
                    for cnt, task in enumerate(task_list):
                        res_test = eval_utils.eval(task, model, test_loaders_twitter[cnt], metric_list[cnt], device)
                        res_test_list.append(res_test)

                    for cnt, task in enumerate(task_list):
                        res_dev = eval_utils.eval(task, model, dev_loaders_twitter[cnt], metric_list[cnt], device)

                        save_flag = False
                        if best_dev_res[cnt] is None:
                            best_dev_res[cnt] = res_dev
                            save_flag = True

                        if task == 'AESC':
                            logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                                res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))
                            tb_writer.add_scalar("aesc_f/dev", res_dev['aesc_f'], t)  # Babu
                            tb_writer.add_scalar("aesc_f/test", res_test_list[cnt]['aesc_f'], t)  # Babu
                            if best_dev_res[cnt]['aesc_f'] < res_dev['aesc_f']:
                                save_flag = True
                                
                        elif task == 'AE':
                            logger.info('DEV  ae_p:{} ae_r:{} ae_f:{}'.format(
                                res_dev['oe_pre'], res_dev['oe_rec'], res_dev['oe_f']))
                            tb_writer.add_scalar("ae_f/dev", res_dev['oe_f'], t)  # Babu
                            tb_writer.add_scalar("ae_f/test", res_test_list[cnt]['oe_f'], t)  # Babu
                            if best_dev_res[cnt]['oe_f'] < res_dev['oe_f']:
                                save_flag = True

                        elif task == 'SC':
                            logger.info('DEV  sc_p:{} sc_r:{} sc_f:{} sc_acc:{}'.format(
                                res_dev['sc_pre'], res_dev['sc_rec'], res_dev['sc_f'], res_dev['sc_acc']))
                            tb_writer.add_scalar("sc_f/dev", res_dev['sc_f'], t)  # Babu
                            tb_writer.add_scalar("sc_acc/dev", res_dev['sc_acc'], t)  # Babu
                            tb_writer.add_scalar("sc_f/test", res_test_list[cnt]['sc_f'], t)  # Babu
                            tb_writer.add_scalar("sc_acc/test", res_test_list[cnt]['sc_acc'], t)  # Babu
                            if best_dev_res[cnt]['sc_acc'] < res_dev['sc_acc']:
                                save_flag = True

                        if args.is_check == 1 and save_flag:
                            best_dev_res[cnt] = res_dev
                            best_dev_test_res_list[cnt] = res_test_list
                            best_dev_epoch[cnt] = epoch + 1
                            best_dev_dataset[cnt] = dataset
                            current_checkpoint_path = os.path.join(checkpoint_path,
                                                                args.check_info, task)
                            model.seq2seq_model.save_pretrained(current_checkpoint_path)
                            print('save model!!!!!!!!!!!')

        epoch += 1
    logger.info("Training complete in: " + str(datetime.now() - start),
                pad=True)
    logger.info('---------------------------')
    logger.info('BEST DEV:-----')
    for cnt, task in enumerate(task_list):
        if task == 'AESC':
            logger.info('BEST DEV {} aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                task, best_dev_res[cnt]['aesc_pre'], best_dev_res[cnt]['aesc_rec'], best_dev_res[cnt]['aesc_f']))
                
        elif task == 'AE':
            logger.info('BEST DEV {} ae_p:{} ae_r:{} ae_f:{}'.format(
                task, best_dev_res[cnt]['oe_pre'], best_dev_res[cnt]['oe_rec'], best_dev_res[cnt]['oe_f']))

        elif task == 'SC':
            logger.info('BEST DEV {} sc_p:{} sc_r:{} sc_f:{} sc_acc:{}'.format(
                task, best_dev_res[cnt]['sc_pre'], best_dev_res[cnt]['sc_rec'], best_dev_res[cnt]['sc_f'], best_dev_res[cnt]['sc_acc']))
    
        logger.info('BEST DEV TEST:-----')
        for cnt_test, task_test in enumerate(task_list):
            if task_test == 'AESC':
                logger.info('BEST DEV {} TEST {} aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                    task, task_test, best_dev_test_res_list[cnt][cnt_test]['aesc_pre'], best_dev_test_res_list[cnt][cnt_test]['aesc_rec'], best_dev_test_res_list[cnt][cnt_test]['aesc_f']))
                    
            elif task_test == 'AE':
                logger.info('BEST DEV {} TEST {} ae_p:{} ae_r:{} ae_f:{}'.format(
                    task, task_test, best_dev_test_res_list[cnt][cnt_test]['oe_pre'], best_dev_test_res_list[cnt][cnt_test]['oe_rec'], best_dev_test_res_list[cnt][cnt_test]['oe_f']))

            elif task_test == 'SC':
                logger.info('BEST DEV {} TEST {} sc_p:{} sc_r:{} sc_f:{} sc_acc:{}'.format(
                    task, task_test, best_dev_test_res_list[cnt][cnt_test]['sc_pre'], best_dev_test_res_list[cnt][cnt_test]['sc_rec'], best_dev_test_res_list[cnt][cnt_test]['sc_f'], best_dev_test_res_list[cnt][cnt_test]['sc_acc']))
        
    logger.info('BEST DEV EPOCH & DATASET:-----')
    for cnt, task in enumerate(task_list):
        logger.info('BEST DEV {}: Epoch {}'.format(task, best_dev_epoch[cnt]))
        logger.info('BEST DEV {}: Dataset {}'.format(task, best_dev_dataset[cnt]))

def parse_args():
    parser = argparse.ArgumentParser()
    # to change
    parser.add_argument('--lamb', default=1, type=float, help='learning rate')
    parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='path to load AESC weights')
    parser.add_argument('--dataset',
                        default='./src/data/jsons/twitter15_info.json',
                        type=str,
                        help='json path to info of twitter dataset')
    parser.add_argument('--MVSA_dataset',
                        default='src/data/jsons/MVSA_with_label_info_by17.json',
                        type=str,
                        help='json path to info of twitter dataset')
    parser.add_argument('--log_dir',
                        default='cotraining',
                        type=str,
                        help='path to output log files, not output to file if not specified')
    # parser.add_argument('--checkpoint_dir',
    #                     default='./',
    #                     type=str,
    #                     help='where to save the checkpoint')
    # 是否启用该数据集
    parser.add_argument('--Twitter_enabled', type=int, default=1, help='Twitter_enabled')
    parser.add_argument('--MVSA_enabled', type=int, default=0, help='MVSA_enabled')
    # 是否启用该任务
    parser.add_argument('--aesc_enabled', type=int, default=1, help='aesc_enabled')
    parser.add_argument('--ae_enabled', type=int, default=1, help='ae_enabled')
    parser.add_argument('--sc_enabled', type=int, default=1, help='sc_enabled')
    
    parser.add_argument('--text_only', default=0, type=int, help='text_only')
    parser.add_argument('--is_check', type=int, default=1, help='start_idx')
    parser.add_argument('--is_sample', type=int, default=0, help='is_sample')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--max_len', type=int, default=30, help='max_len')
    parser.add_argument('--max_len_a', type=float, default=0.6, help='max_len_a')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--check_info', type=str, default='', help='check path to save')
    parser.add_argument('--bart_model', default='facebook/bart-base', type=str, help='bart pretrain model')

    # path
    parser.add_argument('--model_config',
                        default='./config/pretrain_base.json',
                        type=str,
                        help='path to load model config')

    # model
    parser.add_argument('--no_event',
                        dest='use_event',
                        action='store_false',
                        help='not to use event descriptions')
    # parser.add_argument('--no_image',
    #                     dest='use_image',
    #                     action='store_false',
    #                     help='not to use image features')

    # training and evaluation
    parser.add_argument('--epochs',
                        default=40,  # 40
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--num_gen',
                        default=1,
                        type=int,
                        help='number of generated sentence on validation')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument(
        '--validate_loss',
        action='store_true',
        help='compute the validation loss at the end of each epoch')
    parser.add_argument(
        '--validate_score',
        action='store_true',
        help=
        'compute the validation score (BLEU, METEOR, etc.) at the end of each epoch'
    )
    parser.add_argument('--max_img_num',
                        type=int,
                        default=36,
                        help='max number of image feature per data entry')
    parser.add_argument(
        '--lm_max_len',
        type=int,
        default=30,
        help='max number of words for the language modeling per data entry')

    # dropout
    parser.add_argument(
        '--dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the transformer. This overwrites the model config')
    parser.add_argument(
        '--classif_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the classification layers. This overwrites the model config'
    )
    parser.add_argument(
        '--attention_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the attention layers. This overwrites the model config'
    )
    parser.add_argument(
        '--activation_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the activation layers. This overwrites the model config'
    )

    # hardware and performance
    parser.add_argument('--gpu_num',
                        default=1,
                        type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port',
                        type=str,
                        default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,  # 64
                        help='training batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    parser.add_argument('--task', type=str, default='', help='task type')  # 'twitter_sc' 'twitter_ae'
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    
    args = parser.parse_args()
    return args

    # AESC
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='task type')  # 'twitter_sc' 'twitter_ae'
    parser.add_argument('--checkpoint',
                        default='17_finetune_wopp/2023-04-04-13-55-01-lr4e-05-shared_encoder0/AESC',
                        type=str,
                        help='path to load AESC weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_AESC = parser.parse_args()
    
    # AE
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='twitter_ae', help='task')
    parser.add_argument('--checkpoint',
                        default='17_finetune_wopp/2023-04-04-13-55-01-lr4e-05-shared_encoder0/AE',
                        type=str,
                        help='path to load AE weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_AE = parser.parse_args()
    
    # SC
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='twitter_sc', help='task')
    parser.add_argument('--checkpoint',
                        default='17_finetune_wopp/2023-04-04-13-55-01-lr4e-05-shared_encoder0/SC',
                        type=str,
                        help='path to load SC weights')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    args_SC = parser.parse_args()

    return args, args_AESC, args_AE, args_SC


if __name__ == '__main__':
    # args, args_AESC, args_AE, args_SC = parse_args()
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    # mp.spawn(main, args=(args, ), nprocs=args.gpu_num, join=True)
    # main(0, args, args_AESC, args_AE, args_SC)
    main(0, args)

