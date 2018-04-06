import sys
import os
import shutil
import copy
import logging
import pickle
from datetime import datetime
from configparser import ConfigParser, NoOptionError

import numpy as np
import joblib
import yaml

from seg_nets.keras_models import ED_TCN, Dilated_TCN
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
     Y_val) = (val_data_dict['X_val'],
               val_data_dict['Y_val'])

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
    batch_size = int(config['NETWORK']['batch_size'])
    time_steps = int(config['NETWORK']['time_steps'])
    logger.info('will train network with batches of size {}, '
                'where each spectrogram in batch contains {} time steps'
                .format(batch_size, time_steps))

    n_max_iter = int(config['TRAIN']['n_max_iter'])
    logger.info('maximum number of training steps will be {}'
                .format(n_max_iter))

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

    for train_set_dur in TRAIN_SET_DURS:
        for replicate in REPLICATES:
            costs = []
            val_errs = []
            curr_min_err = 1  # i.e. 100%
            err_patience_counter = 0

            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            training_records_dir = ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
            training_records_path = os.path.join(results_dirname,
                                                 training_records_dir)

            checkpoint_filename = ('checkpoint_train_set_dur_'
                                   + str(train_set_dur) +
                                   '_sec_replicate_'
                                   + str(replicate))
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

            import pdb;pdb.set_trace()

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

            # reshape data for network
            batch_spec_rows = len(train_inds) // batch_size

            # this is the original way reshaping was done
            # note that reshaping this way can truncate data set
            X_train_subset = \
                X_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        batch_spec_rows,
                                                                        -1))
            Y_train_subset = \
                Y_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        -1))
            reshape_size = Y_train_subset.ravel().shape[-1]
            diff = train_inds.shape[-1] - reshape_size
            logger.info('Number of time bins after '
                        'reshaping training data: {}.'.format(reshape_size))
            logger.info('Number of time bins less '
                        'than specified {}: {}'.format(train_inds.shape[-1],
                                                       diff))
            logger.info('Difference in seconds: {}'.format(diff * timebin_dur))

            # note that X_train_subset has shape of (batch, time_bins, frequency_bins)
            # so we permute starting indices from the number of time_bins
            # i.e. X_train_subset.shape[1]
            iter_order = np.random.permutation(X_train_subset.shape[1] - time_steps)
            if len(iter_order) > n_max_iter:
                iter_order = iter_order[0:n_max_iter]
            with open(
                    os.path.join(training_records_path,
                                 "iter_order"),
                    'wb') as iter_order_file:
                pickle.dump(iter_order, iter_order_file)

            input_vec_size = X_train_subset.shape[-1]  # number of columns
            logger.debug('input vec size: '.format(input_vec_size))

            (X_val_batch,
             Y_val_batch,
             num_batches_val) = seg_nets.data_utils.reshape_data_for_batching(X_val,
                                                                              Y_val,
                                                                              batch_size,
                                                                              time_steps,
                                                                              input_vec_size)

            # save scaled reshaped data
            scaled_reshaped_data_filename = os.path.join(training_records_path,
                                                         'scaled_reshaped_spects_duration_{}_replicate_{}'
                                                         .format(train_set_dur, replicate))
            scaled_reshaped_data_dict = {'X_train_subset_scaled_reshaped': X_train_subset,
                                         'Y_train_subset_reshaped': Y_train_subset,
                                         'X_val_scaled_batch': X_val_batch,
                                         'Y_val_batch': Y_val_batch}
            joblib.dump(scaled_reshaped_data_dict, scaled_reshaped_data_filename)

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

            networks_config_file = config['NETWORK']['config_file']
            with open(networks_config_file,'r') as yml:
                networks_config = yaml.load(yml)

            models = []
            for model_type, model_config in networks_config['models'].items():
                if model_type == "ED_TCN":
                    model, param_str = ED_TCN(n_nodes=model_config['n_nodes'],
                                              conv_len=model_config['conv_len'],
                                              n_classes=n_syllables,
                                              n_feat=num_freq_bins,
                                              max_len=batch_spec_rows,
                                              causal=model_config['causal'],
                                              activation=model-config['activation'],
                                              return_param_str=True)
                    models.append(model)
                elif model_type = 'Dilated_TCN'

            for model in models:
                model.fit(X_train_m,
                          Y_train_,
                          nb_epoch=nb_epoch,
                          batch_size=8,
                          verbose=1,
                          sample_weight=M_train[:, :, 0])
