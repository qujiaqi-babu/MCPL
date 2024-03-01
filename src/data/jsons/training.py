from datetime import datetime

import numpy as np
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
# from src.utils import TaskType
def refinetune(task_list,  # 3个任务
               epoch,
               model_list,  # 3个模型
               train_loaders,  # 3个dataloader
               optimizer_list,  # 3个优化器
               device,
               args,
               mvsa=False,  # 当前训练数据集是否为使用伪标签的MVSA
               logger=None,
               callback=None,
               log_interval=10,
               tb_writer=None,
               tb_interval=10,
               scaler=None):

    # assert len(task_list) == len(train_loaders)
    # if mvsa:
    #     logger.info('train on mvsa', pad=True)
    # else:
    #     logger.info('train on twitter', pad=True)
    total_step = len(train_loaders[0])
    for model in model_list:
        model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batchs in enumerate(zip(*train_loaders)):
        # Forward pass
        with autocast(enabled=args.amp):
            loss_all = []
            total_loss = 0
            for cnt, task in enumerate(task_list):
                batch = batchs[cnt]
                # print(batch.keys())
                if task == 'AESC':
                    aesc_infos = {key: value for key, value in batch['AESC'].items()}
                    # model.sc_only = False
                    # model.generator.sc_only = False
                    # model.generator.set_new_generator()
                    # model.seq2seq_model.decoder.need_tag = True
                elif task == 'AE':
                    aesc_infos = {key: value for key, value in batch['TWITTER_AE'].items()}
                    # model.sc_only = False
                    # model.generator.sc_only = False
                    # model.generator.set_new_generator()
                    # model.seq2seq_model.decoder.need_tag = False
                elif task == 'SC':
                    aesc_infos = {key: value for key, value in batch['TWITTER_SC'].items()}
                    # model.sc_only = True
                    # model.generator.sc_only = True
                    # model.generator.set_new_generator()
                    # model.seq2seq_model.decoder.need_tag = True
                else:
                    raise ValueError('invalid task')

                with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                    loss = model_list[cnt].forward(
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(  # list[i]为第i张图片的patch feature tensor:(49, 2048)
                            map(lambda x: x.to(device), batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                        aesc_infos=aesc_infos)

                # Backward and optimize
                cur_step = i + 1 + epoch * total_step
                t_step = args.epochs * total_step
                liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
                # 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练。
                # Linner Warmup：学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
                if mvsa:
                    utils.set_lr(optimizer_list[cnt], liner_warm_rate * args.mvsa_lr)
                else:
                    utils.set_lr(optimizer_list[cnt], liner_warm_rate * args.lr)

                optimizer_list[cnt].zero_grad()

                loss.backward()
                utils.clip_gradient(optimizer_list[cnt], args.grad_clip)

                optimizer_list[cnt].step()

                # print(loss.dtype)
                loss_all.append(loss)

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')



def pretrain(task_list,
             epoch,
             model,
             model_sc,
             train_loaders,
             optimizer,
             optimizer_sc,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):

    # assert len(task_list) == len(train_loaders)

    total_step = len(train_loaders[0])
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batchs in enumerate(zip(*train_loaders)):
        # Forward pass
        with autocast(enabled=args.amp):
            loss_all = []
            total_loss = 0
            for cnt, task in enumerate(task_list):
                batch = batchs[cnt]
                # print(batch.keys())
                if task == 'AE':
                    aesc_infos = {
                        key: value
                        for key, value in batch['TWITTER_AE'].items()
                    }
                elif task == 'SC':
                    aesc_infos = {
                        key: value
                        for key, value in batch['TWITTER_SC'].items()
                    }
                else:
                    aesc_infos = {key: value for key, value in batch['AESC'].items()}
                
                if task == 'SC':
                    with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                        loss = model_sc.forward(
                            input_ids=batch['input_ids'].to(device),
                            image_features=list(  # list[i]为第i张图片的patch feature tensor:(196, 2048)
                                map(lambda x: x.to(device), batch['image_features'])),
                            attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                            aesc_infos=aesc_infos)

                    # Backward and optimize
                    cur_step = i + 1 + epoch * total_step
                    t_step = args.epochs * total_step
                    liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
                    # 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练。
                    # Linner Warmup：学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
                    utils.set_lr(optimizer_sc, liner_warm_rate * args.lr)

                    optimizer_sc.zero_grad()

                    loss.backward()
                    utils.clip_gradient(optimizer_sc, args.grad_clip)

                    optimizer_sc.step()
                else:
                    # 修改后的
                    if task == 'AE':
                        model.seq2seq_model.decoder.need_tag = False
                    with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                        loss = model.forward(
                            input_ids=batch['input_ids'].to(device),
                            image_features=list(  # list[i]为第i张图片的patch feature tensor:(196, 2048)
                                map(lambda x: x.to(device), batch['image_features'])),
                            attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                            aesc_infos=aesc_infos)
                    # 修改后的
                    if task == 'AE':
                        model.seq2seq_model.decoder.need_tag = True
                    # Backward and optimize
                    cur_step = i + 1 + epoch * total_step
                    t_step = args.epochs * total_step
                    liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
                    # 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练。
                    # Linner Warmup：学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
                    utils.set_lr(optimizer, liner_warm_rate * args.lr)

                    optimizer.zero_grad()

                    loss.backward()
                    utils.clip_gradient(optimizer, args.grad_clip)

                    optimizer.step()

                # print(loss.dtype)
                loss_all.append(loss)
            
            # print('Epoch [{}/{}], Step [{}/{}]'.format(epoch + 1, args.epochs, i + 1, total_step))
            # for k, v in zip(task_list, loss_all):
            #     print(k + ':', v.item(), end=' ')
            # print()

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')

# mine:finetune by list
def finetune(task_list,
             epoch,
             model_list,
             train_loader_list,
             optimizer_list,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):

    total_step = len(train_loader_list[0])
    for model in model_list:
        model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batchs in enumerate(zip(*train_loader_list)):
        # Forward pass
        loss_all = []
        for cnt, task in enumerate(task_list):
            batch = batchs[cnt]
            # print(batch.keys())
            if task == 'AE':
                aesc_infos = {
                    key: value
                    for key, value in batch['TWITTER_AE'].items()
                }
            elif task == 'SC':
                aesc_infos = {
                    key: value
                    for key, value in batch['TWITTER_SC'].items()
                }
            else:
                aesc_infos = {key: value for key, value in batch['AESC'].items()}
            
            with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                loss = model_list[cnt].forward(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(  # list[i]为第i张图片的patch feature tensor:(196, 2048)
                        map(lambda x: x.to(device), batch['image_features'])),
                    attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                    aesc_infos=aesc_infos)
            
            # Backward and optimize
            cur_step = i + 1 + epoch * total_step
            t_step = args.epochs * total_step
            liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
            # 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练。
            # Linner Warmup：学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
            utils.set_lr(optimizer_list[cnt], liner_warm_rate * args.lr)

            optimizer_list[cnt].zero_grad()

            loss.backward()
            utils.clip_gradient(optimizer_list[cnt], args.grad_clip)

            optimizer_list[cnt].step()

            loss_all.append(loss)
        
        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')

def fine_tune(epoch,
              model,
              train_loader,
              test_loader,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
        with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
            loss = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(  # list[i]为第i张图片的region feature tensor:(36, 2048)
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                aesc_infos=aesc_infos)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # Backward and optimize

        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        # 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练。
        # Linner Warmup：学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
        utils.set_lr(optimizer, liner_warm_rate * args.lr)

        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()
