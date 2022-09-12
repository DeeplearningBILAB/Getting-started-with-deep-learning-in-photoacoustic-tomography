import os
import random
import shutil
import numpy as np
import cv2
from numpy import asarray
from scipy.io import loadmat
from scipy import ndimage
from numpy import savez
from skimage import exposure
from gooey import Gooey, GooeyParser
@Gooey(program_name='Dataset preparation')

def main():
    setting_msg = 'Cleaning and splitting the dataset in to train,test and validation set'
    parser = GooeyParser(description=setting_msg)
    parser.add_argument('input_dir', help='input_dir_path',type=str, widget='DirChooser',default='E:\Project-5\Optimization_dataset')
    parser.add_argument('filename_train',help='training set name' ,type=str,default= 'train_set')
    parser.add_argument('filename_test', help='test set name', type=str, default='test_set')
    parser.add_argument('filename_validation', help='validation set name', type=str, default='valid_set')
    parser.add_argument('choose_split', help='split in to train/test or train/valid/test', choices=['train/test', 'train/valid/test'], default='train/test')
    parser.add_argument('train_set_percentage', help='train(%) ',type=int,  default= 95)
    parser.add_argument('validation_set_percentage', help='validation(%)- select 0 if split is train/test ', type=int, default=0)
    parser.add_argument('test_set_percentage', help='test(%) ', type=int, default=5)
    parser.add_argument('delete_previous', help='delete previously store dataset type True/False', default=True)
    args = parser.parse_args()
    return args.input_dir,args.filename_train,args.filename_validation,args.filename_test,args.choose_split,args.train_set_percentage,args.validation_set_percentage,args.test_set_percentage, args.delete_previous

def dataset_preparation(in_dir, f_name_train, f_name_valid, f_name_test, dataset_percentages, delete_previous  ):

    src_imgs, trgt_imgs = list(), list()
    for parent_folder in os.listdir(in_dir):
        folder_path_parent =os.path.join(in_dir,parent_folder)

        for id_nonid_folder in os.listdir(folder_path_parent):
            if 'input' in id_nonid_folder:
                folder_path_broadimg = os.path.join(folder_path_parent, id_nonid_folder)

                for nonideal_mat in os.listdir(folder_path_broadimg):
                    image = loadmat(os.path.join(folder_path_broadimg, nonideal_mat))
                    image = np.array(image["pa_img"])
                    # image = exposure.equalize_adapthist(image)
                    # image[image<0.1]=0
                    image = exposure.rescale_intensity(image, in_range='image', out_range=(0.0, 1.0))
                    image = image.astype(np.float64)
                    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    image = np.reshape(image, (128, 128, 1))
                    src_imgs.append(image)
            if 'ground_truth' in id_nonid_folder:
                folder_path_idealimg = os.path.join(folder_path_parent, id_nonid_folder)

                for ideal_mat in os.listdir(folder_path_idealimg):
                    image = loadmat(os.path.join(folder_path_idealimg, ideal_mat))
                    image = np.array(image["pa_img"])
                    # image = exposure.equalize_adapthist(image)
                    # image[image<0.1]=0
                    image = exposure.rescale_intensity(image, in_range='image', out_range=(0.0, 1.0))
                    image = image.astype(np.float64)
                    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    image = np.reshape(image, (128, 128, 1))
                    trgt_imgs.append(image)



    images =  list(zip(src_imgs, trgt_imgs))
    random.shuffle(images)
    src_imgs, trgt_imgs = zip(*images)

    if len(dataset_percentages) == 2:
        new_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        elif delete_previous == True:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)

        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        elif delete_previous == True:
            shutil.rmtree(test_dir)
            os.mkdir(test_dir)

        total_num_imgs = len(src_imgs)
        train_percent = dataset_percentages[0]
        test_percent = dataset_percentages[1]
        valid_inputs = (train_percent >= test_percent and train_percent <= 100 and test_percent <= 100 and train_percent > 0 and test_percent > 0 and train_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent//100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of valid input the ratio 90% train, 10% test was used')
        src_imgs_train= src_imgs[0:num_train]
        trgt_imgs_train= trgt_imgs[0:num_train]
        src_imgs_test = src_imgs[num_train:total_num_imgs]
        trgt_imgs_test = trgt_imgs[num_train:total_num_imgs]
        [src_images_train] = [asarray(src_imgs_train)]
        [trgt_images_train] = [asarray(trgt_imgs_train)]
        [src_images_test] = [asarray(src_imgs_test)]
        [trgt_images_test] = [asarray(trgt_imgs_test)]
        file_train = os.path.join(train_dir, f_name_train + '.npz')
        file_test = os.path.join(test_dir, f_name_test + '.npz')
        savez(file_train, src_images_train, trgt_images_train)
        savez(file_test, src_images_test, trgt_images_test)
        print('The images are loaded', src_images_train.shape, trgt_images_train.shape,src_images_test.shape, trgt_images_test.shape)

    elif len(dataset_percentages) == 3:
        # Making Main data folder
        new_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        # Making train subfolder
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        elif delete_previous == True:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)


        # Making val subfolder
        val_dir = os.path.join(new_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)

        elif delete_previous == True:
            shutil.rmtree(val_dir)
            os.mkdir(val_dir)


        # Making test subfolder
        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        elif delete_previous == True:
            shutil.rmtree(test_dir)
            os.mkdir(test_dir)

        total_num_imgs = len(src_imgs)
        train_percent = dataset_percentages[0]
        val_percent = dataset_percentages[1]
        test_percent = dataset_percentages[2]
        valid_inputs = (train_percent >= test_percent and train_percent >= val_percent and train_percent <= 100 and val_percent <= 100 and test_percent <= 100 and train_percent > 0 and val_percent > 0 and test_percent > 0 and
        train_percent + val_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent // 100))
            num_val = int(round(total_num_imgs * val_percent // 100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            num_val = int(round((total_num_imgs - num_train) / 2))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of a valid input the ratio 90% train, 5% val, 5% test was used')
        src_imgs_train = src_imgs[0:num_train]
        trgt_imgs_train = trgt_imgs[0:num_train]
        src_imgs_valid = src_imgs[num_train:num_train+num_val]
        trgt_imgs_valid = trgt_imgs[num_train:num_train+num_val]
        src_imgs_test = src_imgs[num_train+num_val:total_num_imgs]
        trgt_imgs_test = trgt_imgs[num_train+num_val:total_num_imgs]
        [src_images_train] = [asarray(src_imgs_train)]
        [trgt_images_train] = [asarray(trgt_imgs_train)]
        [src_images_valid] = [asarray(src_imgs_valid)]
        [trgt_images_valid] = [asarray(trgt_imgs_valid)]
        [src_images_test] = [asarray(src_imgs_test)]
        [trgt_images_test] = [asarray(trgt_imgs_test)]
        file_train = os.path.join(train_dir, f_name_train + '.npz')
        file_valid = os.path.join(val_dir, f_name_valid + '.npz')
        file_test = os.path.join(test_dir, f_name_test + '.npz')
        savez(file_train, src_images_train, trgt_images_train)
        savez(file_valid, src_images_valid, trgt_images_valid)
        savez(file_test, src_images_test, trgt_images_test)
        print('The images are loaded', src_images_train.shape, trgt_images_train.shape,src_images_valid.shape, trgt_images_valid.shape,src_images_test.shape, trgt_images_test.shape)


if __name__ == "__main__":
    indir, f_name_train, f_name_valid, f_name_test,choose_split, train_per,valid_per,test_per, delete_previous = main()
    if choose_split =='train/test':
        dataset_percentages=(train_per,test_per)
    elif choose_split == 'train/valid/test':
        dataset_percentages = (train_per, valid_per,test_per)

    dataset_preparation(indir,f_name_train,f_name_valid,f_name_test,dataset_percentages,delete_previous)
