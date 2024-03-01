from datetime import datetime
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
# from src.utils import TaskType


def predict(model_AESC=None,
            model_AE=None,
            model_SC=None,
            tokenizer=None,
            dataset=None,
            pred_loader=None,
            device=None,
            logger=None,
            log_dir=None):
    if model_AESC==None or model_AE==None or model_SC==None or tokenizer==None or pred_loader==None:
        print("model and data should not be None")
    
    targetid2mapping = {value:key for key, value in tokenizer.mapping2targetid.items()}

    json_list = []
    target_shift = len(tokenizer.mapping2targetid) + 2
    cnt_generate = 0  # 记录生成的个数
    cnt_first_save = 0  # 记录第一次过滤后的个数
    cnt_second_save = 0  # 记录第二次过滤后的个数
    cnt_final_save = 0  # 记录最终保存的个数

    for sample, output in tqdm(zip(dataset, pred_loader)):  # batch_size为1
        # Forward pass
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
        # logger.info('AESC:{}'.format(predict))

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
        # logger.info('AE:{}'.format(predict))
        
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
        # logger.info('SC:{}'.format(predict))

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


def fine_tune_on_mvsa(epoch,
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
        with autocast(enabled=args.amp):
            loss = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # Backward and optimize

        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        utils.set_lr(optimizer, liner_warm_rate * args.lr)
        tb_writer.add_scalar("loss/train_on_mvsa", loss.item(), cur_step)  # Babu
        loss *= args.lamb  # Babu
        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()


def co_training(task_list,  # 3个任务
               epoch,
               model,  # 1个模型
               train_loaders,  # 3个dataloader
               optimizer,  # 1个优化器
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
    model.train()
    total_loss = 0
    tag = "loss/train_on_mvsa" if mvsa else "loss/train_on_twitter"

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
                    model.sc_only = False
                    model.generator.sc_only = False
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = True
                elif task == 'AE':
                    aesc_infos = {key: value for key, value in batch['TWITTER_AE'].items()}
                    model.sc_only = False
                    model.generator.sc_only = False
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = False
                elif task == 'SC':
                    aesc_infos = {key: value for key, value in batch['TWITTER_SC'].items()}
                    model.sc_only = True
                    model.generator.sc_only = True
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = True
                else:
                    raise ValueError('invalid task')

                with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                    loss = model.forward(
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
                utils.set_lr(optimizer, liner_warm_rate * args.lr)
                tb_writer.add_scalar(tag, loss.item(), cur_step)  # Babu
                
                if mvsa:
                    loss *= args.lamb

                optimizer.zero_grad()

                loss.backward()
                utils.clip_gradient(optimizer, args.grad_clip)

                optimizer.step()

                # print(loss.dtype)
                loss_all.append(loss)

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')


def co_training_curriculum(task_list,  # 3个任务
               epoch,
               model,  # 1个模型
               train_loaders,  # 3个dataloader
               optimizer,  # 1个优化器
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
    model.train()
    total_loss = 0
    tag = "loss/train_on_mvsa" if mvsa else "loss/train_on_twitter"

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
                    model.sc_only = False
                    model.generator.sc_only = False
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = True
                elif task == 'AE':
                    aesc_infos = {key: value for key, value in batch['TWITTER_AE'].items()}
                    model.sc_only = False
                    model.generator.sc_only = False
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = False
                elif task == 'SC':
                    aesc_infos = {key: value for key, value in batch['TWITTER_SC'].items()}
                    model.sc_only = True
                    model.generator.sc_only = True
                    model.generator.set_new_generator()
                    model.seq2seq_model.decoder.need_tag = True
                else:
                    raise ValueError('invalid task')

                with autocast(enabled=args.amp):  # 混合精度训练，减少GPU内存消耗
                    loss = model.forward(
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(  # list[i]为第i张图片的patch feature tensor:(196, 2048)
                            map(lambda x: x.to(device), batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),  # padding部分不计入attention
                        aesc_infos=aesc_infos)

                # Backward and optimize
                cur_step = i + 1 + epoch * total_step
                t_step = args.epochs * total_step

                if mvsa:
                    liner_warm_rate = utils.liner_warmup_cosine_decay(cur_step, t_step, args.warmup)
                    # 学习率从非常小的数值线性增加到预设值之后，然后再余弦衰减。
                else:
                    liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
                    # 学习率从非常小的数值线性增加到预设值之后，然后再线性减小。
                
                if task == 'AESC':
                    rate = cur_step / t_step
                else:
                    rate = 0.5 * ( 1 - cur_step / t_step )

                utils.set_lr(optimizer, liner_warm_rate * args.lr * rate)  # Babu
                tb_writer.add_scalar(tag, loss.item(), cur_step)  # Babu
                
                if mvsa:
                    loss *= args.lamb

                optimizer.zero_grad()

                loss.backward()
                utils.clip_gradient(optimizer, args.grad_clip)

                optimizer.step()

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
             train_loaders,
             optimizer_dict,
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
                if task == 'Sentiment':
                    loss, prelogits = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        })
                else:
                    loss = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        mlm_infos={
                            key: value.to(device)
                            for key, value in batch['MLM'].items()
                        } if 'MLM' in batch else None,
                        mrm_infos={
                            key: value
                            for key, value in batch['MRM'].items()
                        } if 'MRM' in batch else None,
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        } if 'Sentiment' in batch else None,
                        ANP_infos={
                            key: value.to(device)
                            for key, value in batch['ANP'].items()
                        } if 'ANP' in batch else None,
                        ANP_generate_infos={
                            key: value.to(device)
                            for key, value in batch['ANP_generate'].items()
                        } if 'ANP_generate' in batch else None,
                        ae_oe_infos={
                            key: value
                            for key, value in batch['AE_OE'].items()
                        } if 'AE_OE' in batch else None)

                # print(loss.dtype)
                loss_all.append(loss)
                optimizer_dict.zero_grad()

                loss.backward()
                optimizer_dict.step()

            for k, v in zip(task_list, loss_all):
                print(k + ':', v.item(), end=' ')
            print()
        # Backward and optimize

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
        with autocast(enabled=args.amp):
            loss = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # Backward and optimize

        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        utils.set_lr(optimizer, liner_warm_rate * args.lr)
        tb_writer.add_scalar("loss/train", loss.item(), cur_step)  # Babu

        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()
