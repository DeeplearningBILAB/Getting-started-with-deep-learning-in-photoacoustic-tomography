import os
import cv2
import numpy as np
import tensorflow as tf

from scipy.io import savemat
from matplotlib import pyplot as plt

from gooey import Gooey, GooeyParser


@Gooey(program_name='Dataset preparation')
def main():
    setting_msg = 'Simulated Data '
    parser = GooeyParser(description=setting_msg)
    parser.add_argument('test_dir', help='test directory',type=str, widget='FileChooser',default='')
    parser.add_argument('model_file', help='model directory', type=str, widget='FileChooser', default='')
    parser.add_argument('save_dir', help='Directory where the mat files has to be saved', type=str, widget='DirChooser')
    parser.add_argument('num_samples', help='number of samples to plot ',type=int,  default= 3)
    parser.add_argument('im_no', help='range of images ', type=int, default=0)
    parser.add_argument('fig_name', help='model directory', type=str, default='testplot')
    parser.add_argument('save_fig', help='dedo you want to save figure True/False', default=False)
    args = parser.parse_args()
    return args.test_dir,args.model_file, args.save_dir,args.num_samples,args.im_no,args.fig_name,args.save_fig

def data_gen(filename):
    with np.load(filename) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images


def model_load(test_filename,model_file,save_dir,num_samples,im_no,fig_name,save_fig):
    predictions_test = list()
    src_test , tar_test = data_gen(test_filename)
    print(len(src_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((src_test))
    test_dataset = test_dataset.batch(1)
    saved_model = tf.keras.models.load_model(model_file)
    print('Done Loading Best Model ')
    for element in test_dataset.as_numpy_iterator():
        predictions_curr = saved_model.predict(element, steps = 1)
        predictions_test.append(predictions_curr)
    [predictions_test] = [np.asarray(predictions_test)]
    predictions = np.reshape(predictions_test, (predictions_test.shape[0],128, 128))
    src_images = np.reshape(src_test, (src_test.shape[0],128, 128))
    tar_images = np.reshape(tar_test, (tar_test.shape[0],128, 128))
    for i in range(num_samples):
        plt.subplot(3, num_samples, 1 +  i)
        plt.axis('off')
        plt.imshow(src_images[i+im_no],cmap='gist_yarg')
        plt.title('input')
        plt.subplot(3, num_samples, 1 +num_samples+ i)
        plt.axis('off')
        plt.imshow(predictions[i+im_no],cmap='gist_yarg')
        plt.title('predicted')
        plt.subplot(3, num_samples, 1 + num_samples*2  + i)
        plt.axis('off')
        plt.imshow(tar_images[i+im_no],cmap='gist_yarg')
        plt.title('ground truth')
    if save_fig is True:
        plt.savefig(fig_name+'.jpg',dpi=150)
    plt.show()

    for i in range(10):
        predicted_image = cv2.resize(predictions[i + im_no], dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        src_image = cv2.resize(src_images[i + im_no], dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        tar_image = cv2.resize(tar_images[i + im_no], dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        savemat(os.path.join(save_dir, 'src'+ str(i) + '.mat'), {'pa_img': src_image}, appendmat=True)
        savemat(os.path.join(save_dir, 'tar'+str(i) + '.mat'), {'pa_img': tar_image}, appendmat=True)
        savemat(os.path.join(save_dir, str(i) + '.mat'), {'pa_img': predicted_image}, appendmat=True)

if __name__=='__main__':
    test_dir, model_file, save_dir,num_samples, im_no, fig_name, save_fig = main()
    model_load(test_dir, model_file, save_dir, num_samples, im_no, fig_name, save_fig)
