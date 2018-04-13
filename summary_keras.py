import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser
from datetime import datetime

import numpy as np
import joblib
import yaml

from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

import seg_nets.data_utils
import seg_nets.metrics

config_file = sys.argv[1]
if not config_file.endswith('.ini'):
    raise ValueError('{} is not a valid config file, must have .ini extension'
                     .format(config_file))
config = ConfigParser()
config.read(config_file)

results_dirname = config['OUTPUT']['results_dir_made_by_main_script']
if not os.path.isdir(results_dirname):
    raise FileNotFoundError('{} directory is not found.'
                            .format(results_dirname))

timenow = datetime.now().strftime('%y%m%d_%H%M%S')
summary_dirname = os.path.join(results_dirname,
                               'summary_' + timenow)
os.makedirs(summary_dirname)

batch_size = int(config['TRAIN']['batch_size'])
time_steps = int(config['NETWORK']['time_steps'])

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]

num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)
normalize_spectrograms = config.getboolean('TRAIN', 'normalize_spectrograms')

spect_params = {}
for spect_param_name in ['freq_cutoffs', 'thresh']:
    try:
        if spect_param_name == 'freq_cutoffs':
            freq_cutoffs = [float(element)
                            for element in
                            config['SPECTROGRAM']['freq_cutoffs'].split(',')]
            spect_params['freq_cutoffs'] = freq_cutoffs
        elif spect_param_name == 'thresh':
            spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])

    except NoOptionError:
        logger.info('Parameter for computing spectrogram, {}, not specified. '
                    'Will use default.'.format(spect_param_name))
        continue
if spect_params == {}:
    spect_params = None

labelset = list(config['DATA']['labelset'])
skip_files_with_labels_not_in_labelset = config.getboolean(
    'DATA',
    'skip_files_with_labels_not_in_labelset')
labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
with open(labels_mapping_file, 'rb') as labels_map_file_obj:
    labels_mapping = pickle.load(labels_map_file_obj)

train_data_dict_path = config['TRAIN']['train_data_path']
train_data_dict = joblib.load(train_data_dict_path)

(X_train,
Y_train,
train_timebin_dur,
train_spect_params,
train_labels) = (train_data_dict['X_train'],
                 train_data_dict['Y_train'],
                 train_data_dict['timebin_dur'],
                 train_data_dict['spect_params'],
                 train_data_dict['labels'])
X_train = X_train.T

# only get this just to have in summary file if needed
if all(type(labels_el) is str for labels_el in train_labels):
    # when taken from .not.mat files associated with .cbin audio files
    Y_train_labels = ''.join(train_labels)
    Y_train_labels_for_lev = Y_train_labels
elif all(type(labels_el) is np.ndarray for labels_el in train_labels):
    # when taken from annotation.mat supplied with .wav audio files
    Y_train_labels = np.concatenate(train_labels).tolist()
    Y_train_labels_for_lev = ''.join([chr(lbl) for lbl in Y_train_labels])
elif all(type(labels_el) is list for labels_el in train_labels):
    # when taken from annotation.xml supplied with Koumura .wav audio files
    Y_train_labels = [lbl for lbl_list in train_labels for lbl in lbl_list]
    if all([type(lbl) is int for lbl in Y_train_labels]):
        Y_train_labels_for_lev = ''.join([chr(lbl) for lbl in Y_train_labels])
    elif all([type(lbl) is str for lbl in Y_train_labels]):
        Y_train_labels_for_lev = ''.join(Y_train_labels)
    else:
        raise TypeError('Couldn\'t determine type for training labels in {}'
                        .format(type(train_data_dict_path)))
else:
    raise TypeError('Not able to determine type of labels in train data')

print('loading testing data')

test_data_dict_path = config['TRAIN']['test_data_path']
test_data_dict = joblib.load(test_data_dict_path)

# notice data is called `X_test_copy` and `Y_test_copy`
# because main loop below needs a copy of the original
# to normalize and reshape
(X_test_copy,
 Y_test_copy,
 test_timebin_dur,
 files_used,
 test_spect_params,
 test_labels) = (test_data_dict['X_test'],
                 test_data_dict['Y_test'],
                 test_data_dict['timebin_dur'],
                 test_data_dict['filenames'],
                 test_data_dict['spect_params'],
                 test_data_dict['labels'])

assert train_spect_params == test_spect_params
assert train_timebin_dur == test_timebin_dur

# have to transpose X_test so rows are timebins and columns are frequencies
X_test_copy = X_test_copy.T

# used for Levenshtein distance + syllable error rate
if all(type(labels_el) is str for labels_el in test_labels):
    # when taken from .not.mat files associated with .cbin audio files
    Y_test_labels = ''.join(test_labels)
    Y_test_labels_for_lev = Y_test_labels
elif all(type(labels_el) is np.ndarray for labels_el in test_labels):
    # when taken from annotation.mat supplied with .wav audio files
    Y_test_labels = np.concatenate(test_labels).tolist()
    Y_test_labels_for_lev = ''.join([chr(lbl) for lbl in Y_test_labels])
elif all(type(labels_el) is list for labels_el in test_labels):
    # when taken from annotation.xml supplied with Koumura .wav audio files
    Y_test_labels = [lbl for lbl_list in test_labels for lbl in lbl_list]
    if all([type(lbl) is int for lbl in Y_train_labels]):
        Y_test_labels_for_lev = ''.join([chr(lbl) for lbl in Y_train_labels])
    elif all([type(lbl) is str for lbl in Y_train_labels]):
        Y_test_labels_for_lev = ''.join(Y_test_labels)
    else:
        raise TypeError('Couldn\'t determine type for test labels in {}'
                        .format(type(test_data_dict_path)))
else:
    raise TypeError('Not able to determine type of labels in test data')

networks_config_file = config['NETWORK']['config_file']
networks_config_path = config['NETWORK']['config_path']
networks_config_file = os.path.join(networks_config_path,
                                    networks_config_file)
with open(networks_config_file, 'r') as yml:
    networks_config = yaml.load(yml)

err_by_model = {}
for model_name, model_config in networks_config['models'].items():
    err_dict = {}
    # initialize arrays to hold summary results
    err_dict['Y_pred_test_all'] = np.empty((len(TRAIN_SET_DURS),
                                            len(REPLICATES)),
                                           dtype=object)
    err_dict['Y_pred_train_all'] = np.empty((len(TRAIN_SET_DURS),
                                             len(REPLICATES)),
                                            dtype=object)
    err_dict['Y_pred_test_labels_all'] = np.empty((len(TRAIN_SET_DURS),
                                                   len(REPLICATES)),
                                            dtype=object)
    err_dict['Y_pred_train_labels_all'] = np.empty((len(TRAIN_SET_DURS),
                                                    len(REPLICATES)),
                                            dtype=object)
    err_dict['train_err_arr'] = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
    err_dict['test_err_arr'] = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
    err_dict['train_lev_arr'] = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
    err_dict['test_lev_arr'] = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
    err_dict['train_syl_err_arr'] = np.empty((len(TRAIN_SET_DURS),
                                              len(REPLICATES)))
    err_dict['test_syl_err_arr'] = np.empty((len(TRAIN_SET_DURS),
                                             len(REPLICATES)))

    err_by_model[model_name] = err_dict

for dur_ind, train_set_dur in enumerate(TRAIN_SET_DURS):

    for rep_ind, replicate in enumerate(REPLICATES):
        print("getting train and test error for "
              "training set with duration of {} seconds, "
              "replicate {}".format(train_set_dur, replicate))
        training_records_dir = os.path.join(results_dirname,
                                            ('records_for_training_set_with_duration_of_'
                                             + str(train_set_dur) + '_sec_replicate_'
                                             + str(replicate))
                                            )

        train_inds_file = glob(os.path.join(training_records_dir, 'train_inds'))[0]
        with open(os.path.join(train_inds_file), 'rb') as train_inds_file:
            train_inds = pickle.load(train_inds_file)

        # get training set
        Y_train_subset = Y_train[train_inds]
        X_train_subset = X_train[train_inds, :]
        assert Y_train_subset.shape[0] == X_train_subset.shape[0],\
            "mismatch between X and Y train subset shapes"

        binarizer = LabelBinarizer()
        binarizer.fit(Y_train_subset.ravel())

        # Normalize before reshaping to avoid even more convoluted array reshaping.
        # Train spectrograms were already normalized
        # just need to normalize test spects
        if normalize_spectrograms:
            scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                           .format(train_set_dur, replicate))
            spect_scaler = joblib.load(os.path.join(results_dirname, scaler_name))
            X_train_subset = spect_scaler.transform(X_train_subset)
            X_test = spect_scaler.transform(X_test_copy)
        else:
            # get back "un-reshaped" X_test
            X_test = np.copy(X_test_copy)

        # need to get Y_test from copy because it gets reshaped every time through loop
        Y_test = np.copy(Y_test_copy)

        # save scaled spectrograms
        scaled_data_filename = os.path.join(summary_dirname,
                                                 'scaled_spects_duration_{}_replicate_{}'
                                                 .format(train_set_dur, replicate))
        scaled_data_dict = {
            'X_train_subset_scaled': X_train_subset,
            'X_test_scaled': X_test}
        joblib.dump(scaled_data_dict, scaled_data_filename)

        # now that we normalized, we can reshape
        Y_train_subset = binarizer.transform(Y_train_subset)
        (X_train_subset,
         Y_train_subset) = seg_nets.data_utils.reshape_for_input(X_train_subset,
                                                                 Y_train_subset,
                                                                 time_steps)

        Y_test = binarizer.transform(Y_test)
        (X_test,
         Y_test) = seg_nets.data_utils.reshape_for_input(X_test,
                                                         Y_test,
                                                         time_steps)

        scaled_reshaped_data_filename = os.path.join(summary_dirname,
                                            'scaled_reshaped_spects_duration_{}_replicate_{}'
                                            .format(train_set_dur, replicate))
        scaled_reshaped_data_dict = {'X_train_subset_scaled_reshaped': X_train_subset,
                                     'Y_train_subset_reshaped': Y_train_subset,
                                     'X_test_scaled_reshaped': X_test,
                                     'Y_test_reshaped': Y_test}
        joblib.dump(scaled_reshaped_data_dict, scaled_reshaped_data_filename)

        if 'Y_pred_train' in locals():
            del Y_pred_train

        models = []
        for model_name, model_config in networks_config['models'].items():
            print('calculating training set error for {}'
                  .format(model_name))
            training_records_path = os.path.join(results_dirname,
                                                 training_records_dir)
            checkpoint_filename = ('checkpoint_'
                                   + model_name +
                                   '_train_set_dur_'
                                   + str(train_set_dur) +
                                   '_sec_replicate_'
                                   + str(replicate))
            checkpoint_filename = os.path.join(training_records_path,
                                               checkpoint_filename)
            model = load_model(checkpoint_filename)

            Y_pred_train = model.predict(X_train_subset,
                                         batch_size=batch_size,
                                         verbose=1)
            # save softmax probs output by model
            err_by_model[model_name]['Y_pred_train_all'][dur_ind, rep_ind] = Y_pred_train

            # convert from softmax probs to integers
            Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0] * Y_pred_train.shape[1],
                                                Y_pred_train.shape[2])
            Y_pred_train= np.argmax(Y_pred_train, axis=-1)

            Y_train_subset = Y_train[train_inds]  # get back "unreshaped" Y_train_subset
            # get rid of predictions to zero padding that don't matter
            Y_pred_train = Y_pred_train.ravel()[:Y_train_subset.shape[0], np.newaxis]

            train_err = np.sum(Y_pred_train - Y_train_subset != 0) / Y_train_subset.shape[0]
            err_by_model[model_name]['train_err_arr'][dur_ind, rep_ind] = train_err

            print('train error was {}'.format(train_err))

            Y_train_subset_labels = seg_nets.data_utils.convert_timebins_to_labels(Y_train_subset,
                                                                                   labels_mapping)
            Y_pred_train_labels = seg_nets.data_utils.convert_timebins_to_labels(Y_pred_train,
                                                                                 labels_mapping)
            err_by_model[model_name]['Y_pred_train_labels_all'][dur_ind, rep_ind] = Y_pred_train_labels

            if all([type(el) is int for el in Y_train_subset_labels]):
                # if labels are ints instead of str
                # convert to str just to calculate Levenshtein distance
                # and syllable error rate.
                # Let them be weird characters (e.g. '\t') because that doesn't matter
                # for calculating Levenshtein distance / syl err rate
                Y_train_subset_labels = ''.join([chr(el) for el in Y_train_subset_labels])
                Y_pred_train_labels = ''.join([chr(el) for el in Y_pred_train_labels])

            train_lev = seg_nets.metrics.levenshtein(Y_pred_train_labels,
                                                       Y_train_subset_labels)
            err_by_model[model_name]['train_lev_arr'][dur_ind, rep_ind] = train_lev
            print('Levenshtein distance for train set was {}'.format(train_lev))
            train_syl_err_rate = seg_nets.metrics.syllable_error_rate(Y_train_subset_labels,
                                                                        Y_pred_train_labels)
            err_by_model[model_name]['train_syl_err_arr'][dur_ind, rep_ind] = train_syl_err_rate
            print('Syllable error rate for train set was {}'.format(train_syl_err_rate))

            if 'Y_pred_test' in locals():
                del Y_pred_test

            print('calculating test set error')
            Y_pred_test = model.predict(X_test,
                                        batch_size=batch_size,
                                        verbose=1)
            # save softmax probs output by model
            err_by_model[model_name]['Y_pred_test_all'][dur_ind, rep_ind] = Y_pred_test

            # convert from softmax probs to integers
            Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0] * Y_pred_test.shape[1],
                                                Y_pred_test.shape[2])
            Y_pred_test = np.argmax(Y_pred_test, axis=-1)
            # again get rid of zero padding predictions
            Y_pred_test = Y_pred_test.ravel()[:Y_test_copy.shape[0], np.newaxis]
            test_err = np.sum(Y_pred_test - Y_test_copy != 0) / Y_test_copy.shape[0]
            err_by_model[model_name]['test_err_arr'][dur_ind, rep_ind] = test_err
            print('test error was {}'.format(test_err))

            Y_pred_test_labels = seg_nets.data_utils.convert_timebins_to_labels(Y_pred_test,
                                                                                labels_mapping)
            err_by_model[model_name]['Y_pred_test_labels_all'][dur_ind, rep_ind] = Y_pred_test_labels
            if all([type(el) is int for el in Y_pred_test_labels]):
                # if labels are ints instead of str
                # convert to str just to calculate Levenshtein distance
                # and syllable error rate.
                # Let them be weird characters (e.g. '\t') because that doesn't matter
                # for calculating Levenshtein distance / syl err rate
                Y_pred_test_labels = ''.join([chr(el) for el in Y_pred_test_labels])
                # already converted actual Y_test_labels from int to str above,
                # stored in variable `Y_test_labels_for_lev`

            test_lev = seg_nets.metrics.levenshtein(Y_pred_test_labels,
                                                      Y_test_labels_for_lev)
            err_by_model[model_name]['test_lev_arr'][dur_ind, rep_ind] = test_lev
            print('Levenshtein distance for test set was {}'.format(test_lev))
            test_syl_err_rate = seg_nets.metrics.syllable_error_rate(Y_test_labels_for_lev,
                                                                       Y_pred_test_labels)
            print('Syllable error rate for test set was {}'.format(test_syl_err_rate))
            err_by_model[model_name]['test_syl_err_arr'][dur_ind, rep_ind] = test_syl_err_rate

err_by_model['Y_train_labels'] = Y_train_labels,
err_by_model['Y_test_labels'] = Y_test_labels,
err_by_model['train_set_durs'] = TRAIN_SET_DURS

err_by_model_dict_filename = os.path.join(summary_dirname,
                                          'y_preds_and_err_for_train_and_test')
joblib.dump(pred_and_err_dict, pred_err_dict_filename)
