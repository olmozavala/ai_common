import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *
from preproc.constants import NormParams
from pandas import DataFrame

from metrics import dice_coef_loss, real_dice_coef
from os.path import join
import numpy as np

def getAllCallbacksLesion(model_name, early_stopping_func, weights_folder, logs_folder):
    logdir = "{}/run-{}/".format(logs_folder, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.2f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model,monitor=early_stopping_func, save_best_only=True,
                                              mode='max',save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping_func, min_delta=.001, patience=150, mode='max')
    return [logger, save_callback, stop_callback]


def get_all_callbacks(model_name, early_stopping_func, weights_folder, logs_folder):
    logdir = "{}/run-{}/".format(logs_folder, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.5f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model, monitor=early_stopping_func,
                                              save_best_only=True,
                                              save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping_func, min_delta=.0001, patience=200, mode='min')
    return [logger, save_callback, stop_callback]


def runAndSaveModel(model, X, Y, model_name, epochs, is3d, val_per, early_stopping_func) :

    # plot_model(model, to_file='../images/{}.png'.format(model_name), show_shapes=True)

    [logger, save_callback, stop_callback] = get_all_callbacks(model_name, early_stopping_func)

    if is3d:
        model.fit(
            x = X,
            y = Y,
            epochs=epochs,
            shuffle=True,
            batch_size=1,
            verbose=2,
            validation_split=val_per,
            callbacks=[logger, save_callback, stop_callback]
        )
    else:
        model.fit(
            x = X,
            y = Y,
            epochs=epochs,
            shuffle=True,
            batch_size=4,
            verbose=2,
            validation_split=val_per,
            callbacks=[logger]
        )


def split_train_and_test(num_examples, test_percentage):
    """
    Splits a number into training and test randomly
    :param num_examples: int of the number of examples
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idxs = np.arange(num_examples)
    np.random.shuffle(all_samples_idxs)
    test_examples = int(np.ceil(num_examples * test_percentage))
    # Train and validation indexes
    train_val_idxs = all_samples_idxs[0:len(all_samples_idxs) - test_examples]
    test_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples:len(all_samples_idxs)]

    return [train_val_idxs, test_idxs]


def split_train_validation_and_test(num_examples, val_percentage, test_percentage):
    """
    Splits a number into training, validation, and test randomly
    :param num_examples: int of the number of examples
    :param val_percentage: int of the percentage desired for validation
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idxs = np.arange(num_examples)
    np.random.shuffle(all_samples_idxs)
    test_examples = int(np.ceil(num_examples * test_percentage))
    val_examples = int(np.ceil(num_examples * val_percentage))
    # Train and validation indexes
    train_idxs = all_samples_idxs[0:len(all_samples_idxs) - test_examples - val_examples]
    val_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples - val_examples:len(all_samples_idxs) - test_examples]
    test_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples:]
    train_idxs.sort()
    val_idxs.sort()
    test_idxs.sort()

    return [train_idxs, val_idxs, test_idxs]


def trainMultipleModels(dataContainer, imgs_dims, model_name, architectures=['2D','simple', '2DU', '2DSec']):

    optimizers_str = ['sgd']
    #losses = [dice_coef_loss, 'mean_squared_error', 'categorical_crossentropy']
    losses = [dice_coef_loss]
    metrics = [real_dice_coef, 'accuracy']
    epochs = 1000

    # For SGD the hyperams are 1) learning rate 2) decay
    #r = -4*np.random.random(10) # from .0001  to 1 (log)
    #rd = -4*np.random.random(10)# from .0001  to 1 (log)

    #hyperpar = {'sgd': {'lr': 10**r, 'decay': 10**rd}}
    # hyperpar = {'sgd': {'lr': 6*np.random.random(3), 'decay': .01*np.random.random(3), \
    #                     'mom':  .8+.2*np.random.random(3)}}

    hyperpar = {'sgd': {'lr': [.001], 'decay': [.0001], 'mom': [.9]}}

    for optim in optimizers_str:
        if optim == 'sgd':
            for dc in hyperpar.get(optim).get('decay'):
                for mom in hyperpar.get(optim).get('mom'):
                    for lr in hyperpar.get(optim).get('lr'):
                        sgd = SGD(lr=lr, decay=dc, momentum=mom)

                        for loss in losses:
                            if callable(loss):
                                loss_str = loss.__name__
                            else:
                                loss_str = loss

                            for curr_arc in architectures:
                                is3d = False
                                if curr_arc == 'simple':
                                    model = simpleModel(imgs_dims)

                                if curr_arc == '2D':
                                    model = trainModel2D(imgs_dims)

                                if curr_arc == '2DSec':
                                    model = trainModel2DSec(imgs_dims)

                                if curr_arc == '3D_single':
                                    model = getModel_3D_Single(imgs_dims)
                                    is3d = True

                                if curr_arc == '2DU':
                                    model = trainModel2DUNet(imgs_dims)

                                f_mod_name = "{}_Opt_{}_lr_{:2.3f}_mom_{:2.3f}_decay_{:2.4f}_{}_{}".format( \
                                    curr_arc, optim, lr, mom, dc, loss_str, model_name)
                                print("\n Compiling ******** {} ******".format(f_mod_name))
                                model.compile(loss=loss, optimizer=optim, metrics=metrics)

                                print("Fitting .....")
                                runAndSave(model, dataContainer, f_mod_name, epochs, is3d)

    return model


def save_splits(file_name, train_idxs, val_idxs, test_idxs):
    """
    This function saves the training, validation and test indexes. It assumes that there are
    more training examples than validation and test examples. It also uses
    :param file_name:
    :param train_idxs:
    :param val_idxs:
    :param test_idxs:
    :return:
    """
    print("Saving split information...")
    info_splits = DataFrame({F'Train({len(train_ids)})': train_ids})
    info_splits[F'Validation({len(val_ids)})'] = np.nan
    info_splits[F'Validation({len(val_ids)})'][0:len(val_ids)] = val_ids
    info_splits[F'Test({len(test_ids)})'] = np.nan
    info_splits[F'Test({len(test_ids)})'][0:len(test_ids)] = test_ids
    info_splits.to_csv(file_name_splits, index=None)


def save_norm_params(file_name, norm_type, scaler):
    print("Saving normalization parameters....")

    if norm_type == NormParams.min_max:
        file = open(file_name, 'w')
        min_val = scaler.data_min_
        max_val = scaler.data_max_
        scale = scaler.scale_
        range = scaler.data_range_

        file.write(F"Normalization type: {norm_type}, min: {min_val}, max: {max_val}, range: {range}, scale: {scale}")
        file.close()
    else:
        print(F"WARNING! The normalization type {norm_type} is unknown!")


