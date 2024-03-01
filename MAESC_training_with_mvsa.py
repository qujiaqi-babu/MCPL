import argparse
import json
import os
# 使用第x张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from datetime import datetime
from torch import optim
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
from src.model.model import MultiModalBartModelForPretrain
from src.training import fine_tune, fine_tune_on_mvsa
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
from src.model.metrics import AESCSpanMetric
from src.model.generater import SequenceGeneratorModel
import src.eval_utils as eval_utils
import numpy as np
import torch.backends.cudnn as cudnn


def main(rank, args):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tb_writer = None
    add_name = '-epochs' + str(args.epochs)
    # add_name += 'last'

    if args.text_only:
        add_name += '-only_text'
    else:
        add_name += '-multi'
    if args.bart_init == 0:
        add_name += '-random_init'
    if args.checkpoint:
        add_name += '-pretrain_' + args.checkpoint.split('/')[-2]

    add_name += '-lr' + str(args.lr)
    add_name += '-lamb' + str(args.lamb)
    log_dir = os.path.join(args.log_dir, timestamp + add_name)
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp + add_name)

    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=os.path.join('runs', args.log_dir, add_name))  # Babu

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=(rank == 0))

    # make checkpoint dir if not exist
    if args.is_check == 1 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer(args=args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(
            json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    bos_token_id = 0  # 因为是特殊符号
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
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
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
        # model = MultiModalBartModel_AESC(bart_config, args.bart_model,
        #                                  tokenizer, label_ids)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    scaler = GradScaler() if args.amp else None

    epoch = 0
    logger.info('Loading data...')
    collate_aesc = Collator(tokenizer,
                            mlm_enabled=False,
                            senti_enabled=False,
                            ae_enabled=False,
                            oe_enabled=False,
                            aesc_enabled=True,
                            anp_enabled=False,
                            text_only=args.text_only)

    train_dataset = Twitter_Dataset(args.dataset[0][1], split='train')
    mvsa_dataset = MVSA_Dataset_with_label(args.dataset[1][1])

    dev_dataset = Twitter_Dataset(args.dataset[0][1], split='dev')
    test_dataset = Twitter_Dataset(args.dataset[0][1], split='test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              collate_fn=collate_aesc)
    mvsa_loader = DataLoader(dataset=mvsa_dataset,
                              batch_size=args.batch_size*4,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              collate_fn=collate_aesc)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=collate_aesc)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_aesc)

    callback = None
    metric = AESCSpanMetric(eos_token_id,
                            num_labels=len(label_ids),
                            conflict_id=-1)
    model.train()
    start = datetime.now()
    best_dev_res = None
    best_dev_test_res = None
    best_test_res = None
    # res_dev = eval_utils.eval(model, dev_loader, metric, device)
    while epoch < args.epochs * 2:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        if epoch % 2 == 0: 
            logger.info('Train on {}'.format(args.dataset[0][0]))
            fine_tune(epoch=epoch/2,
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    metric=metric,
                    optimizer=optimizer,
                    args=args,
                    device=device,
                    logger=logger,
                    callback=callback,
                    log_interval=1,
                    tb_writer=tb_writer,
                    tb_interval=1,
                    scaler=scaler)
        else:
            logger.info('Train on {}'.format(args.dataset[1][0]))
            fine_tune_on_mvsa(epoch=epoch/2,
                    model=model,
                    train_loader=mvsa_loader,
                    test_loader=test_loader,
                    metric=metric,
                    optimizer=optimizer,
                    args=args,
                    device=device,
                    logger=logger,
                    callback=callback,
                    log_interval=1,
                    tb_writer=tb_writer,
                    tb_interval=1,
                    scaler=scaler)

        print('test!!!!!!!!!!!!!!')
        if (epoch + 1) % args.eval_every == 0:
            # train_dev = eval_utils.eval(model, train_loader, metric, device)
            res_dev = eval_utils.eval(args, model, dev_loader, metric, device)
            res_test = eval_utils.eval(args, model, test_loader, metric, device)

            logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))

            logger.info('TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                res_test['aesc_pre'], res_test['aesc_rec'],
                res_test['aesc_f']))
            
            tb_writer.add_scalar("aesc_p/dev", res_dev['aesc_pre'], epoch + 1)  # Babu
            tb_writer.add_scalar("aesc_r/dev", res_dev['aesc_rec'], epoch + 1)  # Babu
            tb_writer.add_scalar("aesc_f/dev", res_dev['aesc_f'], epoch + 1)  # Babu

            tb_writer.add_scalar("aesc_p/test", res_test['aesc_pre'], epoch + 1)  # Babu
            tb_writer.add_scalar("aesc_r/test", res_test['aesc_rec'], epoch + 1)  # Babu
            tb_writer.add_scalar("aesc_f/test", res_test['aesc_f'], epoch + 1)  # Babu

            save_flag = False
            if best_dev_res is None:
                best_dev_res = res_dev
                best_dev_test_res = res_test

            else:
                if best_dev_res['aesc_f'] < res_dev['aesc_f']:
                    best_dev_res = res_dev
                    best_dev_test_res = res_test

            if best_test_res is None:
                best_test_res = res_test
                save_flag = True
            else:
                if best_test_res['aesc_f'] < res_test['aesc_f']:
                    best_test_res = res_test
                    save_flag = True

            if args.is_check == 1 and save_flag:
                current_checkpoint_path = os.path.join(checkpoint_path,
                                                       args.check_info)
                model.seq2seq_model.save_pretrained(current_checkpoint_path)
                print('save model!!!!!!!!!!!')
        epoch += 1
    logger.info("Training complete in: " + str(datetime.now() - start),
                pad=True)
    logger.info('---------------------------')
    logger.info('BEST DEV:-----')
    logger.info('BEST DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_res['aesc_pre'], best_dev_res['aesc_rec'],
        best_dev_res['aesc_f']))

    logger.info('BEST DEV TEST:-----')
    logger.info('BEST DEV--TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_test_res['aesc_pre'], best_dev_test_res['aesc_rec'],
        best_dev_test_res['aesc_f']))

    logger.info('BEST TEST:-----')
    logger.info('BEST TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_test_res['aesc_pre'], best_test_res['aesc_rec'],
        best_test_res['aesc_f']))

    # if not args.cpu:
    #     cleanup_process()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        metavar=('DATASET_NAME', 'DATASET_PATH'),
                        required=True,
                        help='')
    # required

    parser.add_argument('--checkpoint_dir',
                        required=True,
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='facebook/bart-base',
                        type=str,
                        help='bart pretrain model')
    # path
    parser.add_argument(
        '--log_dir',
        default=None,
        type=str,
        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config',
                        default=None,
                        type=str,
                        help='path to load model config')
    parser.add_argument('--text_only',
                        default=False,
                        type=bool,
                        help='if only input the text')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='name or path to load weights')
    parser.add_argument('--lr_decay_every',
                        default=4,
                        type=int,
                        help='lr_decay_every')
    parser.add_argument('--lr_decay_ratio',
                        default=0.8,
                        type=float,
                        help='lr_decay_ratio')
    # training and evaluation
    parser.add_argument('--epochs',
                        default=35,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lamb', default=1, type=float, help='learning rate')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
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
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
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
                        default=16,
                        help='training batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--max_len', type=int, default=10, help='max_len')
    parser.add_argument('--max_len_a',
                        type=float,
                        default=0.6,
                        help='max_len_a')

    parser.add_argument('--bart_init',
                        type=int,
                        default=1,
                        help='use bart_init or not')

    parser.add_argument('--check_info',
                        type=str,
                        default='',
                        help='check path to save')
    parser.add_argument('--is_check',
                        type=int,
                        default=0,
                        help='save the model or not')
    parser.add_argument('--task', type=str, default='', help='task type')
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')

    return args


if __name__ == '__main__':
    args = parse_args()

    # mp.spawn(main, args=(args, ), nprocs=args.gpu_num, join=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    main(0, args)
