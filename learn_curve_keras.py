import sys
import os
import shutil
import copy
import logging
import pickle
from datetime import datetime
from configparser import ConfigParser, NoOptionError
import time

import numpy as np
import joblib
import yaml
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from seg_nets.keras_models import ed_tcn_output_size
from seg_nets.keras_models import ED_TCN, Dilated_TCN, CNN_biLSTM
import seg_nets.data_utils


if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
    if not os.path.isfile(config_file):
        raise FileNotFoundError('config file {} is not found'
                                .format(config_file))
    config = ConfigParser()
    config.read(config_file)

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_section('OUTPUT'):
        if config.has_option('OUTPUT', 'results_dir'):
            output_dir = config['OUTPUT']['results_dir']
            results_dirname = os.path.join(output_dir,
                                           'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.makedirs(results_dirname)
    # copy config file into results dir now that we've made the dir
    shutil.copy(config_file, results_dirname)

    logfile_name = os.path.join(results_dirname,
                                'logfile_from_running_main_' + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Logging results to {}'.format(results_dirname))
    logger.info('Using config file: {}'.format(config_file))

    train_data_dict_path = config['TRAIN']['train_data_path']
    logger.info('Loading training data from {}'.format(
        os.path.dirname(
            train_data_dict_path)))
    train_data_dict = joblib.load(train_data_dict_path)
    labels_mapping = train_data_dict['labels_mapping']

    # require user to specify parameters for spectrogram
    # instead of having defaults (as was here previously)
    # helps ensure we don't mix up different params
    spect_params = {}
    spect_params['fft_size'] = int(config['SPECTROGRAM']['fft_size'])
    spect_params['step_size'] = int(config['SPECTROGRAM']['step_size'])
    spect_params['freq_cutoffs'] = [float(element)
                                    for element in
                                    config['SPECTROGRAM']['freq_cutoffs']
                                        .split(',')]
    if config.has_option('SPECTROGRAM', 'thresh'):
        spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
    if config.has_option('SPECTROGRAM', 'transform_type'):
        spect_params['transform_type'] = config['SPECTROGRAM']['transform_type']

    if train_data_dict['spect_params'] != spect_params:
        raise ValueError('Spectrogram parameters in config file '
                         'do not match parameters specified in training data_dict.\n'
                         'Config file is: {}\n'
                         'Data dict is: {}.'.format(config_file,
                                                    train_data_dict_path))

    # save copy of labels_mapping in results directory
    labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
    with open(labels_mapping_file, 'wb') as labels_map_file_obj:
        pickle.dump(labels_mapping, labels_map_file_obj)

    # copy training data to results dir so we have it stored with results
    logger.info('copying {} to {}'.format(train_data_dict_path,
                                          results_dirname))
    shutil.copy(train_data_dict_path, results_dirname)

    (X_train,
     Y_train,
     X_train_spect_ID_vector,
     timebin_dur,
     files_used) = (train_data_dict['X_train'],
                    train_data_dict['Y_train'],
                    train_data_dict['spect_ID_vector'],
                    train_data_dict['timebin_dur'],
                    train_data_dict['filenames'])

    logger.info('Size of each timebin in spectrogram, in seconds: {}'
                .format(timebin_dur))
    # dump filenames to a text file
    # to be consistent with what the matlab helper function does
    files_used_filename = os.path.join(results_dirname, 'training_filenames')
    with open(files_used_filename, 'w') as files_used_fileobj:
        files_used_fileobj.write('\n'.join(files_used))

    total_train_set_duration = float(config['DATA']['total_train_set_duration'])
    dur_diff = np.abs((X_train.shape[-1] * timebin_dur) - total_train_set_duration)
    if dur_diff > 1.0:
        raise ValueError('Duration of X_train in seconds from train_data_dict '
                         'is more than one second different from '
                         'duration specified in config file.\n'
                         'train_data_dict: {}\n'
                         'config file: {}'
                         .format(train_data_dict_path, config_file))
    logger.info('Total duration of training set (in s): {}'
                .format(total_train_set_duration))

    TRAIN_SET_DURS = [int(element)
                      for element in
                      config['TRAIN']['train_set_durs'].split(',')]
    max_train_set_dur = np.max(TRAIN_SET_DURS)

    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(TRAIN_SET_DURS))

    # transpose X_train, so rows are timebins and columns are frequency bins
    # because networks expect this orientation for input
    X_train = X_train.T
    # for initializing networks below, need number of frequency bins
    num_freq_bins = X_train.shape[-1]  # i.e. number of columns
    # save training set to get training accuracy in summary.py
    joblib.dump(X_train, os.path.join(results_dirname, 'X_train'))
    joblib.dump(Y_train, os.path.join(results_dirname, 'Y_train'))

    num_replicates = int(config['TRAIN']['replicates'])
    REPLICATES = range(num_replicates)
    logger.info('will replicate training {} times for each duration of training set'
                .format(num_replicates))

    val_data_dict_path = config['TRAIN']['val_data_path']
    val_data_dict = joblib.load(val_data_dict_path)
    (X_val,
     Y_val,
     X_val_spect_ID_vector) = (val_data_dict['X_val'],
                               val_data_dict['Y_val'],
                               val_data_dict['spect_ID_vector'],)

    if val_data_dict['spect_params'] != spect_params:
        raise ValueError('Spectrogram parameters in config file '
                         'do not match parameters specified in validation data_dict.\n'
                         'Config file is: {}\n'
                         'Data dict is: {}.'.format(config_file,
                                                    val_data_dict_path))
    #####################################################
    # note that we 'transpose' the spectrogram          #
    # so that rows are time and columns are frequencies #
    #####################################################
    X_val = X_val.T

    val_dur = int(config['TRAIN']['val_dur'])
    logger.info('will measure error on validation set '
                'of duration {} seconds'.format(val_dur))
    val_inds = seg_nets.data_utils.get_inds_for_dur(X_val_spect_ID_vector,
                                                    Y_val,
                                                    labels_mapping,
                                                    val_dur,
                                                    timebin_dur)
    X_val = X_val[val_inds, :]
    Y_val = Y_val[val_inds]
    joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
    joblib.dump(Y_val, os.path.join(results_dirname, 'Y_val'))

    val_error_step = int(config['TRAIN']['val_error_step'])
    logger.info('will measure error on validation set '
                'every {} steps of training'.format(val_error_step))
    checkpoint_step = int(config['TRAIN']['checkpoint_step'])
    logger.info('will save a checkpoint file '
                'every {} steps of training'.format(checkpoint_step))
    save_only_single_checkpoint_file = config.getboolean('TRAIN',
                                                         'save_only_single_checkpoint_file')
    if save_only_single_checkpoint_file:
        logger.info('save_only_single_checkpoint_file = True\n'
                    'will save only one checkpoint file'
                    'and overwrite every {} steps of training'.format(checkpoint_step))
    else:
        logger.info('save_only_single_checkpoint_file = False\n'
                    'will save a separate checkpoint file '
                    'every {} steps of training'.format(checkpoint_step))

    patience = config['TRAIN']['patience']
    try:
        patience = int(patience)
    except ValueError:
        if patience == 'None':
            patience = None
        else:
            raise TypeError('patience must be an int or None, but'
                            'is {} and parsed as type {}'
                            .format(patience, type(patience)))
    logger.info('\'patience\' is set to: {}'.format(patience))

    # set params used for sending data to graph in batches
    batch_size = int(config['TRAIN']['batch_size'])
    time_steps = int(config['NETWORK']['time_steps'])
    logger.info('will train network with batches of size {}, '
                'where each spectrogram in batch contains {} time steps'
                .format(batch_size, time_steps))


    nb_epoch = int(config['TRAIN']['nb_epoch'])
    logger.info('number of training epochs will be {}'
                .format(nb_epoch))

    normalize_spectrograms = config.getboolean('TRAIN', 'normalize_spectrograms')
    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')
        # need a copy of X_val when we normalize it below
        X_val_copy = copy.deepcopy(X_val)

    use_train_subsets_from_previous_run = config.getboolean(
        'TRAIN', 'use_train_subsets_from_previous_run')
    if use_train_subsets_from_previous_run:
        try:
            previous_run_path = config['TRAIN']['previous_run_path']
        except NoOptionError:
            raise ('In config.file {}, '
                   'use_train_subsets_from_previous_run = Yes, but'
                   'no previous_run_path option was found.\n'
                   'Please add previous_run_path to config file.'
                   .format(config_file))

    label_pad_value = max(labels_mapping.values()) + 1
    logger.info('will pad label vectors with value: {}'
                .format(label_pad_value))

    for train_set_dur in TRAIN_SET_DURS:
        for replicate in REPLICATES:

            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            training_records_dir = ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
            training_records_path = os.path.join(results_dirname,
                                                 training_records_dir)

            if not os.path.isdir(training_records_path):
                os.makedirs(training_records_path)

            if use_train_subsets_from_previous_run:
                train_inds_path = os.path.join(previous_run_path,
                                               training_records_dir,
                                               'train_inds')
                with open(train_inds_path, 'rb') as f:
                    train_inds = pickle.load(f)
            else:
                train_inds = seg_nets.data_utils.get_inds_for_dur(X_train_spect_ID_vector,
                                                                  Y_train,
                                                                  labels_mapping,
                                                                  train_set_dur,
                                                                  timebin_dur)
            with open(os.path.join(training_records_path, 'train_inds'),
                      'wb') as train_inds_file:
                pickle.dump(train_inds, train_inds_file)
            X_train_subset = X_train[train_inds, :]
            Y_train_subset = Y_train[train_inds]
            spect_ID_subset = X_train_spect_ID_vector[train_inds]

            if normalize_spectrograms:
                spect_scaler = seg_nets.data_utils.SpectScaler()
                X_train_subset = spect_scaler.fit_transform(X_train_subset)
                logger.info('normalizing validation set to match training set')
                X_val = spect_scaler.transform(X_val_copy)
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                joblib.dump(spect_scaler,
                            os.path.join(results_dirname, scaler_name))

            scaled_data_filename = os.path.join(training_records_path,
                                                'scaled_spects_duration_{}_replicate_{}'
                                                .format(train_set_dur, replicate))
            scaled_data_dict = {'X_train_subset_scaled': X_train_subset,
                                'X_val_scaled': X_val,
                                'Y_train_subset': Y_train_subset}
            joblib.dump(scaled_data_dict, scaled_data_filename)

            input_vec_size = X_train_subset.shape[-1]  # number of columns
            logger.debug('input vec size: '.format(input_vec_size))


            spect_ID_vec_tmp = X_val_spect_ID_vector + (spect_ID_subset[-1] + 1)
            spect_ID_vec_tmp = np.concatenate((spect_ID_subset,
                                               spect_ID_vec_tmp))
            max_len = np.max(
                np.unique(spect_ID_vec_tmp, return_counts=True)[1])

            networks_config_file = config['NETWORK']['config_file']
            networks_config_path = config['NETWORK']['config_path']
            networks_config_file = os.path.join(networks_config_path,
                                                networks_config_file)
            with open(networks_config_file,'r') as yml:
                networks_config = yaml.load(yml)

            if any(model_config['type']=="ED_TCN"
                   for _, model_config in networks_config['models'].items()):
                iter = 0
                for _, model_config in networks_config['models'].items():
                    if model_config['type']=="ED_TCN":
                        n_nodes = model_config['n_nodes']
                        if time_steps != ed_tcn_output_size(time_steps, n_nodes):
                            raise ValueError("value for time_steps, {}, does not"
                                             "result in same output size from"
                                             "from ED_TCN decoder output.\nTest "
                                             "value with ed_tcn_output_size "
                                             "function.")

            binarizer = LabelBinarizer()
            binarizer.fit(np.concatenate((Y_train_subset.ravel(),
                                          Y_val.ravel())))
            Y_train_subset = binarizer.transform(Y_train_subset)
            (X_train_subset,
            Y_train_subset) = seg_nets.data_utils.window_data(X_train_subset,
                                                             Y_train_subset,
                                                             time_steps)
            seed = int(round(time.time() * 1000))
            seg_nets.data_utils.seedyshuffle(X_train_subset,
                                             Y_train_subset,
                                             seed)

            Y_val_batch = binarizer.transform(Y_val)
            (X_val_batch,
            Y_val_batch) = seg_nets.data_utils.window_data(X_val,
                                                          Y_val_batch,
                                                          time_steps)
            seed = int(round(time.time() * 1000))
            seg_nets.data_utils.seedyshuffle(X_val_batch,
                                             Y_val_batch,
                                             seed)
            val_data = (X_val_batch, Y_val_batch)

            # n_syllables, i.e., number of label classes to predict
            # Note that mapping includes label for silent gap b/t syllables
            # Error checking code to ensure that it is in fact a consecutive
            # series of integers from 0 to n, so we don't predict classes that
            # don't exist
            if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
                raise ValueError('Labels mapping does not map to a consecutive'
                                 'series of integers from 0 to n (where 0 is the '
                                 'silent gap label and n is the number of syllable'
                                 'labels).')
            n_syllables = len(labels_mapping)
            logger.debug('n_syllables: '.format(n_syllables))

            models = []
            for model_name, model_config in networks_config['models'].items():
                if model_config['type'] == "ED_TCN":
                    if Y_train_subset.shape[1] != ed_tcn_output_size(X_train_subset.shape[1],
                                                                     model_config['n_nodes']):
                        raise ValueError('ED_TCN output does not equal input, '
                                         'check whether max_len was adjusted to '
                                         'appropriate length so that output equals'
                                         'input after max pooling then upsampling.')
                    model = ED_TCN(n_nodes=model_config['n_nodes'],
                                   conv_len=model_config['conv_len'],
                                   n_classes=n_syllables,
                                   n_feat=num_freq_bins,
                                   max_len=time_steps,
                                   causal=model_config['causal'],
                                   activation=model_config['activation'],
                                   optimizer=model_config['optimizer'],
                                   loss=model_config['loss'])

                elif model_config['type'] == 'Dilated_TCN':
                    model = Dilated_TCN(num_feat=num_freq_bins,
                                        num_classes=n_syllables,
                                        nb_filters=model_config['nb_filters'],
                                        dilation_depth=model_config[
                                            'dilation_depth'],
                                        nb_stacks=model_config['nb_stacks'],
                                        max_len=time_steps,
                                        causal=model_config['causal'],
                                        optimizer=model_config['optimizer'],
                                        loss=model_config['loss'])

                elif model_config['type'] == 'CNN_biLSTM':
                    model = CNN_biLSTM(n_classes=n_syllables,
                                       n_feat=num_freq_bins,
                                       max_len=time_steps,
                                       n_filters_by_layer=model_config['n_filters_by_layer'],
                                       loss=model_config['loss'],
                                       optimizer=model_config['optimizer'])

                model_dict = {'name': model_name,
                              'type': model_config['type'],
                              'obj': model}
                models.append(model_dict)

            for model_dict in models:
                logger.info('training model: {}'.format(model_dict['name']))
                logger.info('model type: {}'.format(model_dict['type']))
                checkpoint_filename = ('checkpoint_'
                                       + model_dict['name'] +
                                       '_train_set_dur_'
                                       + str(train_set_dur) +
                                       '_sec_replicate_'
                                       + str(replicate))
                checkpoint_filename = os.path.join(training_records_path,
                                                   checkpoint_filename)
                checkpointer = ModelCheckpoint(checkpoint_filename,
                                               monitor='val_acc',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto', period=1)
                earlystopper = EarlyStopping(monitor='val_loss',
                                             min_delta=0.001,
                                             patience=patience,
                                             verbose=1,
                                             mode='auto')
                log_dir = os.path.join(training_records_path,
                                       'logs',
                                       model_dict['name'])
                if not os.path.isdir(log_dir):
                    os.makedirs(log_dir)
                tensorboarder = TensorBoard(log_dir=log_dir,
                                            histogram_freq=0,
                                            batch_size=1,
                                            write_graph=False,
                                            write_grads=False,
                                            write_images=False,
                                            update_freq='batch')
                tic = time.time()
                history = model_dict['obj'].fit(X_train_subset,
                                                Y_train_subset,
                                                epochs=nb_epoch,
                                                batch_size=batch_size,
                                                verbose=1,
                                                shuffle=True,
                                                validation_data=val_data,
                                                callbacks=[checkpointer,
                                                           earlystopper,
                                                           tensorboarder,
                                                           ])
                toc = time.time()
                logger.info('training start: {}'.format(tic))
                logger.info('training stop: {}'.format(toc))
                logger.info('total training time: {}'.format(toc - tic))

                history_filename = os.path.join(training_records_path,
                                                model_dict['name']
                                                + '_history')
                joblib.dump(history.history,
                            history_filename)