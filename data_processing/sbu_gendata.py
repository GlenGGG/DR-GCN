# This code is based on https://github.com/xrenaa/SBU_Kinect_dataset_process

import urllib
import os
import time
import zipfile
import shutil
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import random
import pickle
#import requests


def unzip(sourceFile, targetPath):
    '''
    :param sourceFile: path to zip files
    :param targetPath: target path
    :return:
    '''
    file = zipfile.ZipFile(sourceFile, 'r')
    file.extractall(targetPath)
    print('success to unzip file!')


def unzipSourceFiles(source='./source', target='./unzip'):

    for i in range(1, 22):
        filename = "Set%02d"%i
        path = os.path.join(source, filename+".zip")
        print(path)
        target_path = os.path.join(target, str(i))
        unzip(path, target_path)


def deleteJunks(target='./unzip'):
    for i in range(1, 22):
        target_path = os.path.join(target, str(i))
        del_path = os.path.join(target_path, "__MACOSX")
        shutil.rmtree(del_path)
        print("deleted junks ",i)


def getSkeletonInfo(target='./unzip'):
    print(target)
    mainFile = sorted(os.listdir(target))
    print(mainFile)

    total = []
    labels = []
    actor_idxs = []
    actor_idx = -1
    for i in range(1, 22):
        filename = '%02d' % i
        mainFile = os.path.join(target, filename)
        subFile = os.listdir(mainFile)[0]
        sub_path = os.path.join(mainFile, subFile)
        cats = sorted(os.listdir(sub_path))[1::]
        actor_idx = i-1
        for cat in cats:
            label = int(cat)-1
            cat_path = os.path.join(sub_path, cat)
            nums = sorted(os.listdir(cat_path))
            if nums[0] == '.DS_Store':
                nums = nums[1::]
            for num in nums:
                list_1 = []
                list_2 = []
                one = []
                num_path = os.path.join(cat_path, num)
                txt_path = os.path.join(num_path, "skeleton_pos.txt")
                with open(txt_path) as f:
                    data = f.readlines()
                for row in data:
                    posture = row
                    pose_1 = []
                    pose_2 = []
                    posture_data = [x.strip() for x in posture.split(',')]

                    joint_info = {}
                    for i, n in enumerate(range(1, len(posture_data), 3)):
                        joint_info[i + 1] = [float(posture_data[n]),
                                             float(posture_data[n + 1]),
                                             float(posture_data[n + 2])]
                    person_1 = {k: joint_info[k] for k in range(1, 16, 1)}
                    person_2 = {k-15: joint_info[k] for k in range(16, 31, 1)}

                    for key, value in person_1.items():
                        pose_1.append(value[0:3])
                    array = np.array(pose_1)
                    list_1.append(array)

                    for key, value in person_2.items():
                        pose_2.append(value[0:3])
                    array = np.array(pose_2)
                    list_2.append(array)

                list_1 = np.array(list_1)
                list_2 = np.array(list_2)
                one.append(list_1)
                one.append(list_2)
                labels.append(label)
                total.append(one)
                actor_idxs.append(actor_idx)
    return total,labels,actor_idxs


def padZeros(total,labels,actor_idxs):
    count = 0
    interpolate = []
    max_mx = 0
    for i in range(282):
        slice = np.array(total[i])
        if slice.shape[1] > 35:
            count += 1
            if slice.shape[1] > max_mx:
                max_mx = slice.shape[1]
        temp = []
        for person in range(2):
            pose = slice[person]
            tensor_1 = torch.tensor(pose)
            tensor_1 = tensor_1.permute(1, 2, 0)
            tensor_1 = tensor_1.numpy()
            mx = tensor_1.shape[2]
            tensor_1 = np.tile(tensor_1, [1, 1, 100//(tensor_1.shape[2])+1])
            tensor_1 = torch.from_numpy(tensor_1)
            tensor_1 = tensor_1[:, :, 0:100]
            tensor_1[:, :, mx:100] = 0
            target_1 = tensor_1
            target_1 = target_1.permute(2, 0, 1)
            temp.append(target_1.numpy())
        interpolate.append(np.array(temp))
    interpolate = np.array(interpolate)
    return interpolate

def saveData(interpolate,labels,actor_idxs, save_dir):
    interpolate_list = []
    labels_list = []
    actor_idxs_list = []
    data_zip = list(zip(interpolate, labels, actor_idxs))
    random.shuffle(data_zip)
    interpolate_list[:], labels_list[:], actor_idxs_list[:] = zip(*data_zip)

    c = np.array(interpolate_list[0:len(interpolate_list)])
    ct = torch.from_numpy(c)
    N, M, T, V, C = ct.shape
    ct = ct.contiguous().permute(0, 4, 2, 3, 1).view(N, C, T, V, M)
    c = ct.numpy().astype(np.float32)
    l = np.array(labels_list[0:len(labels_list)])
    sample_name = str(l)

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    np.save("{}/total_data.npy".format(save_dir), c)
    with open('{}/total_label.pkl'.format(save_dir), 'wb') as f:
        pickle.dump((list(actor_idxs_list), list(labels_list)), f)
    print("saved total_data")

    k_fold_idxs = [
        [1, 9, 15, 19],
        [5, 7, 10, 16],
        [2, 3, 20, 21],
        [4, 6, 8, 11],
        [12, 13, 14, 17, 18]]
    # 5_fold_idxs=[[2,3],[4,5]]
    idxs = [[]for i in range(5)]
    labels_list_fold = [[]for i in range(5)]
    actor_idxs_list_fold = [[]for i in range(5)]
    ii = 0
    for fold_idxs in k_fold_idxs:
        for i in range(c.shape[0]):
            if actor_idxs_list[i] in fold_idxs:
                idxs[ii].append(i)
                labels_list_fold[ii].append(labels_list[i])
                actor_idxs_list_fold[ii].append(actor_idxs_list[i])
        ii = ii+1
    # labels_list_fold=[[],[],[],[],[]]
    # labels_list_fold[1].append(1)
    train_labels_list = []
    train_actor_idxs_list = []
    train_idxs = []
    val_actor_idxs_list = []
    val_labels_list = []
    val_idxs = []

    for i in range(5):
        for j in range(5):
            if j != i:
                train_actor_idxs_list.extend(actor_idxs_list_fold[j])
                train_labels_list.extend(labels_list_fold[j])
                train_idxs.extend(idxs[j])
            else:
                val_actor_idxs_list.extend(actor_idxs_list_fold[j])
                val_labels_list.extend(labels_list_fold[j])
                val_idxs.extend(idxs[j])

        train_data = list(zip(train_idxs, train_labels_list, train_actor_idxs_list))
        random.shuffle(train_data)
        train_idxs[:], train_labels_list[:], train_actor_idxs_list[:] = zip(*train_data)

        train_data_np = c[train_idxs]
        print(train_data_np.dtype)
        np.save("{}/train_data_{}.npy".format(save_dir, i), train_data_np)
        with open('{}/train_label_{}.pkl'.format(save_dir, i), 'wb') as f:
            pickle.dump((list(train_actor_idxs_list), list(train_labels_list)), f)

        val_data = list(zip(val_idxs, val_labels_list, val_actor_idxs_list))
        random.shuffle(val_data)
        val_idxs[:], val_labels_list[:], val_actor_idxs_list[:] = zip(*val_data)
        val_data_np = c[val_idxs]
        print(val_data_np.shape)
        np.save("{}/val_data_{}.npy".format(save_dir, i), val_data_np)
        with open('{}/val_label_{}.pkl'.format(save_dir, i), 'wb') as f:
            pickle.dump((list(val_actor_idxs_list), list(val_labels_list)), f)
    print("saved to {}".format(save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBU Data Converter.')
    parser.add_argument('--data_path',
                        default='data/SBU/source')
    parser.add_argument('--out_folder', default='data/SBU')
    arg = parser.parse_args()
    source=arg.data_path
    save_dir=arg.out_folder
    target_dir=os.path.join(source,"../unzip")
    print(target_dir)
    unzipSourceFiles(source)
    deleteJunks()
    total,labels,actor_idxs=getSkeletonInfo(target_dir)
    interpolate=padZeros(total,labels,actor_idxs)
    saveData(interpolate, labels, actor_idxs,save_dir)
