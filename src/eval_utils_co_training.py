import torch
import torch.nn as nn


def eval(task, model, loader, metric, device):
    model.eval()
    if task == 'AESC':
        model.sc_only = False
        model.generator.sc_only = False
        model.generator.set_new_generator()
        model.seq2seq_model.decoder.need_tag = True
    elif task == 'AE':
        model.sc_only = False
        model.generator.sc_only = False
        model.generator.set_new_generator()
        model.seq2seq_model.decoder.need_tag = False
    elif task == 'SC':
        model.sc_only = True
        model.generator.sc_only = True
        model.generator.set_new_generator()
        model.seq2seq_model.decoder.need_tag = True
    else:
        raise ValueError('invalid task')
    
    task_to_key = {'AESC':'AESC', 'AE':'TWITTER_AE', 'SC':'TWITTER_SC'}
    for i, batch in enumerate(loader):
        # Forward pass
        aesc_infos = {key: value for key, value in batch[task_to_key[task]].items()}
        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos)

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res
