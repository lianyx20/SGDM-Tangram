import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
import random
import csv
import torch.nn.functional as F
import pandas as pd
import json

# 2. 自定义数据集
class TangramDataset(Dataset):
    def __init__(self, image_paths, img_coco_list, preprocess):
        self.image_paths = image_paths
        self.img_coco_list = img_coco_list  # 每个image对应多个text的列表
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        # img_coco = Image.open(self.img_coco_list[idx]).convert("RGB")
        # img_coco = self.preprocess(img_coco)
        # 随机选择一个文本标注
        # texts = np.random.choice(texts)
        return image, self.img_coco_list[idx]

def get_unique_and_indices(lst):
    unique_values = []
    indices_map = {}

    for idx, val in enumerate(lst):
        if val not in indices_map:
            unique_values.append(val)
            indices_map[val] = []
        indices_map[val].append(idx)

    return unique_values, indices_map

def eval_clip(clip,dataloader,device,tokenizer):
    # dataloader should give different imgs, not every img-text pair
    clip.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    n_recall = 10
    n_recallexp = 10
    total_loss = 0
    total_loss_comb = 0
    total_loss_constra = 0
    total_loss_mse = 0
    total_simi = 0
    top1_correct = 0
    top2_correct = 0
    alpha = 10
    with torch.no_grad():
        for batch_idx, (images, img_coco) in enumerate(dataloader):
            
            coco_features = find_embed(img_coco,mode='df')
            images = images.to(device)
            # 计算特征和logits
            image_features = model.encode_image(images)
            logits = (image_features @ coco_features.T) * model.logit_scale.exp()
            # 对比损失
            labels = torch.arange(len(images)).to(device)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            total_loss += loss

            loss_contrastive, loss_mse = loss_cos_mse(image_features,coco_features,logits,target=1.0)
            loss = loss_contrastive + alpha*loss_mse
            total_loss_comb += loss
            total_loss_constra += loss_contrastive
            total_loss_mse += loss_mse

            # similarity
            simi_text2img = torch.cosine_similarity(image_features, coco_features, dim=1)
            total_simi += simi_text2img.mean()

            ## recall out of 10
            total = len(img_coco)

            for expi in range(n_recallexp):
                # get recall samples
                indices = random.sample(range(total), n_recall)  # 随机挑选 n 个不重复索引
                image_recall = image_features[indices]
                text_recall = coco_features[indices]

                # Compute similarity (text-to-image retrieval)
                x_norm = F.normalize(image_recall, dim=1)
                y_norm = F.normalize(text_recall, dim=1)
                simi_recall = x_norm @ y_norm.T

                # For each text, get top-k image indices
                for i in range(n_recall):
                    topk = simi_recall[i].topk(2).indices.cpu().numpy()
                    if i == topk[0]:
                        top1_correct += 1
                    if i in topk:
                        top2_correct += 1# Normalize
        
        average_loss = total_loss / (batch_idx+1)
        average_loss1 = total_loss_comb / (batch_idx+1)
        average_loss2 = total_loss_constra / (batch_idx+1)
        average_loss3 = total_loss_mse / (batch_idx+1)

        average_simi = total_simi / (batch_idx+1)
        recall_1 = top1_correct / ((batch_idx+1)*n_recall*n_recallexp)
        recall_2 = top2_correct / ((batch_idx+1)*n_recall*n_recallexp)

        return average_loss, average_simi, recall_1, recall_2, average_loss1, average_loss2, average_loss3

def data2loader(images_train, coco_train, train_transform, batch_size=16):
    img_unique, indices_map = get_unique_and_indices(images_train)

    coco_list = []
    for i in range(len(img_unique)):
        indices = indices_map[img_unique[i]]
        # coco_list.append([coco_train[j] for j in indices])
        coco_list.append(coco_train[indices[-1]])

    dataset = TangramDataset(img_unique, coco_list, train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def loss_cos_mse(img_f, text_f, logits, alpha=0.5, target=1.0):

    labels = torch.arange(len(logits)).to(device)
    loss_contrastive = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2

    simi = F.cosine_similarity(img_f,text_f,dim=-1)
    loss_mse = F.mse_loss(simi,torch.ones_like(simi)*target)

    # return loss_contrastive + alpha*loss_mse
    return loss_contrastive ,loss_mse

def eeg_unique_img(imglist):
    keyls = []
    for i in range(len(imglist)):
        keyname = imglist[i][67:-4]
        keyls.append(keyname)
    keyls = list(set(keyls))
    return keyls

def text2real(text,df_full,reallst):
    result = []
    for texti in text:
        t = df_full[df_full["anno_whole"]==texti].index.tolist()
        idx = t[0]
        result.append(reallst[idx])
    return result

def find_embed(img_coco,mode='df'):
    # if mode=='df':
    #     idx = []
    #     for i in range(len(images)):
    #         idx.append(tangfeat['files'].index(images[i]))
    #     idx = np.array(idx)
    #     img_embed = tangfeat['clipfeat'][idx]
    # elif mode=='eeg':
    #     idx = []
    #     for i in range(len(images)):
    #         t = images[i]
    #         idx.append(tangfeat['files'].index(images[i]))
    #     idx = np.array(idx)
    #     img_embed = tangfeat['clipfeat'][idx]
    
    idx = []
    for i in range(len(img_coco)):
        idx.append(realfeat['files'].index(img_coco[i]))
    idx = np.array(idx)
    coco_embed = realfeat['clipfeat'][idx]

    return coco_embed


if __name__ == "__main__":
    # 1. 加载模型和预处理
    model_type = 'ViT-H-14'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model.to(device)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=180),
        feature_extractor,
    ])

    df_tangram_full = pd.read_csv(f"/home/nncc/lyx/project/EEG_Image_decode-main/myEEG/full_forcontext.csv")
    # coco_dir = "variables/tangram_coco_match_full.json"
    # with open(coco_dir,"r") as f:
    #     coco = json.load(f)

    # realfeat = torch.load("variables/controlsd_imgfeat.pt")
    realfeat = torch.load("variables/controlsd_imgfeat_ind5.pt")
    tangfeat = torch.load("variables/tang_imgfeat.pt")

    # 示例数据
    from eegdatasets_tang_clip2 import EEGDataset
    sub = 'sub-01'
    data_path = "/home/nncc/lyx/project/EEG_Image_decode-main/myEEG/whiten/Preprocessed_data_250Hz"

    # save_dir = f'dataset_features/{sub}-multilabel'
    # train_dataset = EEGDataset(data_path, subjects=[sub], train=True, multilabel=True, featpth=save_dir)

    ## multilabel
    # save_dir = f'dataset_features/{sub}-filteranno'
    # train_dataset = EEGDataset(data_path, subjects=[sub], train=True, multilabel=True, featpth=save_dir, textmode='whole')
    # test_dataset = EEGDataset(data_path, subjects=[sub], train=False, multilabel=True, featpth=save_dir, textmode='whole')
    
    ## singlelabel
    save_dir = f'dataset_features/{sub}-filteranno'
    train_dataset = EEGDataset(data_path, subjects=[sub], train=True, multilabel=False, featpth=save_dir, textmode='picture')
    test_dataset = EEGDataset(data_path, subjects=[sub], train=False, multilabel=False, featpth=save_dir, textmode='picture')
    
    train_img = eeg_unique_img(train_dataset.img)
    test_img = eeg_unique_img(test_dataset.img)

    train_wholetext = [anno[16:] for anno in train_dataset.text]
    test_wholetext = [anno[16:] for anno in test_dataset.text]

    realsd_dir = f'generated_imgs/anno2img_control/prompt_no'
    keyls = []
    for i in range(len(train_dataset.img)):
        keyname = train_dataset.img[i][67:-4]
        keyls.append(keyname)
    # real_train = [f"{realsd_dir}/{keys}_real.png" for keys in keyls]
    real_train = [f"{realsd_dir}/{keys}_3.png" for keys in keyls]
    keyls = []
    for i in range(len(test_dataset.img)):
        keyname = test_dataset.img[i][67:-4]
        keyls.append(keyname)
    # real_test = [f"{realsd_dir}/{keys}_real.png" for keys in keyls]
    real_test = [f"{realsd_dir}/{keys}_3.png" for keys in keyls]

    full_pth = f'/home/nncc/lyx/project/EEG_Image_decode-main/myEEG'
    df_full = pd.read_csv(f'{full_pth}/filter_anno.csv')
    keys = df_full['key']
    keys_unique = list(dict.fromkeys(keys))

    # coco_pth = f'/home/nncc/lyx/project/EEG_Image_decode-main/myEEG/coco/val2017'
    images_full = []
    img_coco_list = []
    texts_full = []

    ####### single anno, full
    # for i in ind_arr:
    #     tangname = df_full['key'].iloc[i]
    #     texti = df_full['anno_whole'].iloc[i]
    #     # texts_full.append(df_full['anno_whole'].iloc[i])
    #     texts_full.append(f'This picture is {texti}')
    #     images_full.append(f'{full_pth}/KILOGRAM/expdataset/whole/{tangname}.png')

    #     imgcoco = f"{realsd_dir}/{tangname}_real.png"
    #     img_coco_list.append(imgcoco)
    
    ####### single anno
    # for i in ind_arr:
    #     tangname = df_full['key'].iloc[i]
    #     if tangname not in (train_img+test_img):
    #         texti = df_full['anno_whole'].iloc[i]
    #         # texts_full.append(df_full['anno_whole'].iloc[i])
    #         texts_full.append(f'This picture is {texti}')
    #         images_full.append(f'{full_pth}/KILOGRAM/expdataset/whole/{tangname}.png')

    #         imgcoco = f"{realsd_dir}/{tangname}_real.png"
    #         img_coco_list.append(imgcoco)

    ####### multi anno
    for i in range(df_full.shape[0]):
        tangname = df_full['key'].iloc[i]
        ind_5 = i%5
        if tangname not in (train_img+test_img):
            if i % 5 in [1,2,3]:
                texti = df_full['anno_whole'].iloc[i]
                # texts_full.append(df_full['anno_whole'].iloc[i])
                texts_full.append(f'This picture is {texti}')
                images_full.append(f'{full_pth}/KILOGRAM/expdataset/whole/{tangname}.png')

                imgcoco = f"{realsd_dir}/{tangname}_{ind_5}.png"
                img_coco_list.append(imgcoco)
        elif i % 5 in [1,2,4]:
            texti = df_full['anno_whole'].iloc[i]
            # texts_full.append(df_full['anno_whole'].iloc[i])
            texts_full.append(f'This picture is {texti}')

            images_full.append(f'{full_pth}/KILOGRAM/expdataset/whole/{tangname}.png')
            imgcoco = f"{realsd_dir}/{tangname}_{ind_5}.png"
            img_coco_list.append(imgcoco)

    ######### multi anno, full
    # for i in range(df_full.shape[0]):
    #     tangname = df_full['key'].iloc[i]
    #     if df_full['id_anno'].iloc[i]>0: # split test and balance number
    #         images_full.append(f'{full_pth}/KILOGRAM/expdataset/whole/{tangname}.png')
    #         texts_full.append(df_full['anno_whole'].iloc[i])

    dataset = TangramDataset(images_full, img_coco_list, train_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. 冻结所有参数（默认冻结）
    for param in model.parameters():
        param.requires_grad = False

    # # 3. 仅解冻视觉编码器的最后N层（此处解冻最后4层Transformer块）
    # # In fact, I found 32*ResidualAttentionBlock
    # visual_layers_to_train = [
    #     'transformer.resblocks.20',
    #     'transformer.resblocks.21', 
    #     'transformer.resblocks.22',
    #     'transformer.resblocks.23',
    #     'ln_post'  # 层归一化
    # ]
    # for name, param in model.visual.named_parameters():
    #     if any(layer in name for layer in visual_layers_to_train):
    #         param.requires_grad = True

    n_grad = 3
    for block in model.visual.transformer.resblocks[-n_grad:]:
        for param in block.parameters():
            param.requires_grad = True

    # 4. 强制冻结文本投影矩阵（即使文本编码器已冻结）
    model.text_projection.requires_grad_(False)

    # 3. 优化器和损失
    # 3. 优化器配置（修正点
    # optimizer = torch.optim.AdamW([
    #     {"params": model.visual.parameters(), "lr": 1e-6},          # 视觉编码器
    #     # {"params": [model.text_projection], "lr": 5e-5},            # 文本投影矩阵
    #     # {"params": model.transformer.parameters(), "lr": 1e-5}      # 文本编码器（可选解冻）
    # ], weight_decay=0.01)
    # 6. 优化器（仅训练需梯度的参数）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        # weight_decay=0.01
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    alpha = 500 # 500/100/10

    # 4. 训练循环
    results = []
    save_dir = f"/home/nncc/lyx/project/EEG_Image_decode-main/Generation/fintune_ckpts/CLIP/sub-01-07093"
    for epoch in range(20):
        model.train()
        for images, img_coco in tqdm(dataloader):

            coco_features = find_embed(img_coco,mode='df')
            images = images.to(device)
            # 计算特征和logits
            image_features = model.encode_image(images)
            logits = (image_features @ coco_features.T) * model.logit_scale.exp()

            # 对比损失
            # labels = torch.arange(len(images)).to(device)
            # loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2

            # contrastive+mse
            loss_contrastive, loss_mse = loss_cos_mse(image_features,coco_features,logits,target=1.0)
            loss = loss_contrastive + alpha*loss_mse

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # save
        if (epoch+1)%10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/clip_tangram_finetuned_{epoch+1}.pth")

        # eval
        if (epoch+1)%1 == 0:
            model.eval()
            # train eval
            dl = data2loader(images_full, img_coco_list, train_transform, batch_size=32)
            average_loss_tr, average_simi_tr, recall_1_tr, recall_2_tr, avg_loss_comb1, avg_loss_cons1, avg_loss_mse1 = eval_clip(model,dl,device,tokenizer)
            # test eval
            dl = data2loader(test_dataset.img+train_dataset.img, real_test+real_train, train_transform, batch_size=32)
            average_loss_te, average_simi_te, recall_1_te, recall_2_te, avg_loss_comb2, avg_loss_cons2, avg_loss_mse2 = eval_clip(model,dl,device,tokenizer)

            epoch_results = {
                "epoch": epoch + 1,
                "test_loss": average_loss_te.item(),
                "test_simi": average_simi_te.item(),
                "test_recall1": recall_1_te,
                "test_recall2": recall_2_te,
                "test_loss_comb": avg_loss_comb2.item(),
                "test_loss_cons": avg_loss_cons2.item(),
                "test_loss_mse": avg_loss_mse2.item(),

                "train_loss": average_loss_tr.item(),
                "train_simi": average_simi_tr.item(),
                "train_recall1": recall_1_tr,
                "train_recall2": recall_2_tr,
                "train_loss_comb": avg_loss_comb1.item(),
                "train_loss_cons": avg_loss_cons1.item(),
                "train_loss_mse": avg_loss_mse1.item(),
                }
            results.append(epoch_results)

            print(f"Eval Epoch {epoch+1}, test Loss: {average_loss_te:.4f}, test simi: {average_simi_te:.4f}, test_recall@1:{recall_1_te:.4f}, test_recall@2:{recall_2_te:.4f}, loss_comb:{avg_loss_comb2:.4f},loss_cons:{avg_loss_cons2:.4f},loss_mse:{avg_loss_mse2:.4f}")
            print(f"Eval Epoch {epoch+1}, train Loss: {average_loss_tr:.4f}, train simi: {average_simi_tr:.4f}, train_recall@1:{recall_1_tr:.4f}, train_recall@2:{recall_2_tr:.4f}, loss_comb:{avg_loss_comb1:.4f},loss_cons:{avg_loss_cons1:.4f},loss_mse:{avg_loss_mse1:.4f}")


    # 5. 保存模型
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/clip_tangram_finetuned.pth")
    torch.save(optimizer.state_dict(), f"{save_dir}/opti_tangram_finetuned.pth")

    results_file = f"{save_dir}/eval.csv"

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        print(f'Results saved to {results_file}')