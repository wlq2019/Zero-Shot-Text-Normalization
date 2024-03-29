# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     train
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/30
   Software:      PyCharm
'''
import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoader import PunctuationDataset, PunctuationDataset_student,collate_fn,collate_fn_student,PunctuationDataset_dl
from model import BERTForPunctuator
from torch.utils.tensorboard import SummaryWriter
import numpy as np

'''
模型训练
    1. 初始化参数
    2. 初始化数据集，构建dataloader
    3. 构造模型
    4. 构造损失函数
    5. 构造优化器
    6. 迭代训练
        1）清空梯度
        2）计算预测
        3）计算loss
        4）反向传播
        5）参数更新
    7. 存数ckp
    8. 验证模型
'''

parser = argparse.ArgumentParser()

parser.add_argument("--train-set", default="lao_train.txt", help="train dataset path")
parser.add_argument("--train-set-student", default="lao_train.txt", help="train student dataset path")
parser.add_argument("--valid-set", default="lao_test.txt", help="valid dataset path")
parser.add_argument("--label-vocab", default="label.dict.tsv", help="label vocabulary path")
parser.add_argument("--label-size", default=66, help="label dimension")
parser.add_argument("--lr", default=5e-5, type=float, help="learn rate")
parser.add_argument("--batch-size", default=16, help="batch size")
parser.add_argument("--epoch", default=24, help="train times")
parser.add_argument("--device", default="cuda", help="whether use gpu or not")
parser.add_argument("--ckp_teacher", default="checkpoint_TN_teacher/epoch19.pt", help="where to save the checkpoints")
parser.add_argument("--ckp", default="./checkpoint_TN_student_20230618_lao_baseline", help="where to save the checkpoints")

parser.add_argument("--ckp-nums", default=4, help="how checkpoints to hold at the same time")
parser.add_argument("--tb", default="./tb_student_20230618_lao_baseline", help="where the tensorboard saved")
parser.add_argument("--seed", default=1, help="random seed")

args = parser.parse_args()


def get_teacher_predictions():
    """
    Gets predictions by the same method as the zero-shot pipeline but with DataParallel & more efficient batching
    """
    device = torch.device(args.device)

    print("Loading Data...")
    teacher_dataset = PunctuationDataset_dl(input_path=args.train_set_student, label_vocab_path=args.label_vocab)
    teacher_dataloader = DataLoader(teacher_dataset, batch_size=args.batch_size, collate_fn=collate_fn)


    # build model
    print("Building Model...")
    model_teacher = BERTForPunctuator(args.label_size, device)

    # load ckp
    checkpoint_teacher = torch.load(args.ckp_teacher, map_location=device)
    model_teacher.load_state_dict(checkpoint_teacher["model"])

    # move to device
    model_teacher = model_teacher.to(device)

    # set eval mode
    model_teacher.eval()

    print(model_teacher)




   

    preds_all = []
    for sentences, labels in tqdm(teacher_dataloader, desc="[soft lables]"):
        # move data to device
        sentences = sentences.to(device)
        # labels = labels.to(device)

        # forward
        with torch.no_grad():
            preds = model_teacher(sentences)

        soft_teacher = preds.cpu().numpy()
        
        preds_all.append(soft_teacher)

    return preds_all






if __name__ == '__main__':
    '''训练模型'''
    # choose device
    device = torch.device(args.device)

    # set fixed seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    loss_fn2 = torch.nn.MSELoss(reduction='mean')

    preds_all_batched = get_teacher_predictions()
    preds_all = []
    for elem in preds_all_batched:
        for el in elem:
            preds_all.append(el)


    # build dataloader
    print("Loading Data...")
    train_dataset_student = PunctuationDataset_student(input_path=args.train_set, label_vocab_path=args.label_vocab, preds_all= preds_all)
    valid_dataset = PunctuationDataset(input_path=args.valid_set, label_vocab_path=args.label_vocab)
    train_dataloader = DataLoader(train_dataset_student, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_student)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # build model
    print("Building model...")
    model = BERTForPunctuator(args.label_size, device)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    # print(loss_func)
    print(optimizer)

    # move model to device
    model = model.to(device)

    # check if there's checkpoints
    if os.path.exists(args.ckp):
        # 存在checkpoint文件夹
        ckps = [os.path.join(args.ckp, file) for file in os.listdir(args.ckp)]
        if len(ckps) > 0:
            continue_train = input(f"Found {len(ckps)} checkpoints, [c]ontinue the training or [r]emove them all:\n")
            if continue_train == "c":
                checkpoint = torch.load(max(ckps, key=os.path.getctime), map_location=device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                previous_epoch = checkpoint['epoch']
            elif continue_train == "r":
                # 删除所有已保存模型
                [os.remove(file) for file in ckps]
                previous_epoch = 0
    else:
        previous_epoch = 0

    # initiate tensorboard
    tb_writer_train = SummaryWriter(os.path.join(args.tb, "train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.tb, "valid"))

    batch = 0
    for epoch in range(1, args.epoch):
        # set train mode
        model.train()

        epoch += previous_epoch
        train_epoch_loss = 0
        for sentences, soft_labels,WFST_labels  in tqdm(train_dataloader, desc=f"[Epoch {epoch}]"):
            # move data to device
            sentences = sentences.to(device)
            soft_labels = soft_labels.to(device)
            WFST_labels = WFST_labels.to(device)

            # zero gradient
            model.zero_grad()

            # forward
            try:
                preds = model(sentences)
            except Exception as e:
                print(f"Skipping this batch because {e}")
                continue

            # calculate loss
            loss_WFST = loss_func(preds.view(-1, 66), WFST_labels.view(-1))
            loss = loss_fn2(preds.float(), soft_labels.float())

            origin_loss_weight = min((batch * 0.006)**2,1)

            wfst_loss_weight = max(1-(batch * 0.006)**2,0.1)

            # loss_all = wfst_loss_weight * loss_WFST +  origin_loss_weight * loss
            loss_all = loss

            # write loss to tb
            tb_writer_train.add_scalar("loss", loss_all.item(), batch)

            # backward
            loss_all.backward()

            # update parameters
            optimizer.step()

            batch += 1
            train_epoch_loss += loss_all.item()

            # for debug
            # if batch == 5:
            #     break

        print(f"Last batch loss: {loss_all.item()}")
        print(f"Starting validing...")
        # a = 0
        valid_epoch_loss = 0
        for sentences, labels  in tqdm(valid_dataloader, desc="[Validing]"):
            sentences = sentences.to(device)
            labels = labels.to(device)

            model.eval()

            with torch.no_grad():
                try:
                    preds = model(sentences)
                except Exception as e:
                    print(f"Skipping this batch because {e}")
                    continue

                # calculate loss
                loss = loss_func(preds.view(-1, 66), labels.view(-1))

            valid_epoch_loss += loss.item()

            # a += 1

            # for debug
            # if a == 20:
            #     break

        # output valid and train result
        train_ppl = math.exp(train_epoch_loss / len(train_dataloader))
        valid_ppl = math.exp(valid_epoch_loss / len(valid_dataloader))
        tb_writer_train.add_scalar("PPL", train_ppl, epoch)
        tb_writer_valid.add_scalar("PPL", valid_ppl, epoch)
        print(f"Train PPL: {train_ppl: 7.3f}")
        print(f"Valid PPL: {valid_ppl: 7.3f}")

        # save checkpoint
        print("Saving checkpoint...")
        if not os.path.exists(args.ckp):
            os.mkdir(args.ckp)
        if len(os.listdir(args.ckp)) >= args.ckp_nums:
            files = [os.path.join(args.ckp, file) for file in os.listdir(args.ckp)]
            os.remove(min(files, key=os.path.getctime))
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, os.path.join(args.ckp, f"epoch{epoch}.pt"))
        print("Saved!")

