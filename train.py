import argparse
import keras
from src.dataloader import Srinivasan_2014
from src.model import OpticNet, resnet50, mobilenetv2
import time
import keras.backend as K
import gc
from src.utils import callback_for_training
from src.visualize import plot_loss_acc

def train(data_dir, logdir, input_size, batch_size, weights, epoch, pre_trained_model,snapshot_name):
    

    train_generator, validation_generator = Srinivasan_2014(batch_size, input_size, data_dir)
    num_of_classes = 3
    train_size = 2916
    test_size = 315
    # Clear any outstanding net or memory    
    K.clear_session()
    gc.collect()

    # Calculate the starting time
    start_time = time.time()

    # Callbacks for model saving, adaptive learning rate
    cb = callback_for_training(tf_log_dir_name=logdir,snapshot_name=snapshot_name)


    # Loading the model
    if pre_trained_model=='OpticNet71':
        model = OpticNet(input_size,num_of_classes)
    elif pre_trained_model=='ResNet50':
        model = resnet50(input_size,num_of_classes)
    elif pre_trained_model=='MobileNetV2':
        model = mobilenetv2(input_size,num_of_classes)

    # Training the model
    history = model.fit_generator(train_generator, shuffle=True, steps_per_epoch=train_size //batch_size, validation_data=validation_generator, validation_steps= test_size//batch_size, epochs=epoch, verbose=1, callbacks=cb)

    end_time = time.time()

    print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))

    # Saving the final model
    if snapshot_name == None :
        model.save('Joint-Attention-OpticNet.h5')
    elif pre_trained_model=='MobileNetV2':
        model.save('Joint-Attention-MobileNetV2.h5')
    elif pre_trained_model=='ResNet50':
        model.save('Joint-Attention-ResNet50.h5')
    else :    
        model.save(snapshot_name+'.h5')
    
    plot_loss_acc(history,snapshot_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--weights', type=str,default=None, help='Resuming training from previous weights')
    parser.add_argument('--model',type=str, default='OpticNet71',help='Pretrained weights for transfer learning',choices=['OpticNet71','ResNet50',
                                 'MobileNetV2'])
    parser.add_argument('--snapshot_name',type=str, default=None, help='Name the saved snapshot')
    args = parser.parse_args()
    train(args.datadir, args.logdir, args.input_dim, args.batch, args.weights, args.epoch, args.model, args.snapshot_name)
