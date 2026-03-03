# %%------------ Import libraries and initialise
# import nest_asyncio
# nest_asyncio.apply()

# data importing / formatting
import json
import os
import pprint
from pathlib import Path

# data
import numpy as np
import pandas as pd
import copy

# plottinng
import matplotlib.pyplot as plt
import seaborn as sns

# # stats bits
import statsmodels.api as sm
# from statsmodels.formula.api import ols

# saving
import csv
import pickle
from scipy.optimize import curve_fit
from scipy.stats import linregress

experiment_types = ['LVLS', 'LVHS', 'HVLS', 'HVHS']
# os.chdir('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/pain_control_git/paper_submission/experiment1/preprocessing')
os.chdir(
    '/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/elife_revision/preprocessing')

plot_trans_input = False  # Plot transformed input series


# %%------------ Functions for pre-processing
def load_json(file_name):
    f = open(file_name)
    data = json.load(f)
    participant_id = Path(file_name).stem[12:]
    return data, participant_id


def get_data(participant_data_tuple):
    participant_data = participant_data_tuple[0]
    participant_id = participant_data_tuple[1]
    order = participant_data['ORDER']
    sens = participant_data['Sens']
    threshold = participant_data['Threshold']
    sequences = participant_data['sequences']

    for sequence in sequences:
        if len(sequence) == 9:
            if sequence['name'] == 'LVLS':
                LVLS_s = sequence['s']
                LVLS_ms = sequence['ms']
                LVLS_VAS = sequence['VAS']
                LVLS_CP = sequence['CP']
            elif sequence['name'] == 'LVHS':
                LVHS_s = sequence['s']
                LVHS_ms = sequence['ms']
                LVHS_VAS = sequence['VAS']
                LVHS_CP = sequence['CP']
            elif sequence['name'] == 'HVLS':
                HVLS_s = sequence['s']
                HVLS_ms = sequence['ms']
                HVLS_VAS = sequence['VAS']
                HVLS_CP = sequence['CP']
            elif sequence['name'] == 'HVHS':
                HVHS_s = sequence['s']
                HVHS_ms = sequence['ms']
                HVHS_VAS = sequence['VAS']
                HVHS_CP = sequence['CP']
    participant_data_dict = {'participant_id': participant_id,
                             'Order': order,
                             # 'LVLS': {'s': LVLS_s, 'ms': LVLS_ms, 'VAS': LVLS_VAS, 'CP': LVLS_CP},
                             # 'LVHS': {'s': LVHS_s, 'ms': LVHS_ms, 'VAS': LVHS_VAS, 'CP': LVHS_CP},
                             # 'HVLS': {'s': HVLS_s, 'ms': HVLS_ms, 'VAS': HVLS_VAS, 'CP': HVLS_CP},
                             # 'HVHS': {'s': HVHS_s, 'ms': HVHS_ms, 'VAS': HVHS_VAS, 'CP': HVHS_CP}

                             'LVLS': {'s': HVLS_s, 'ms': HVLS_ms, 'VAS': HVLS_VAS, 'CP': HVLS_CP},
                             'LVHS': {'s': HVHS_s, 'ms': HVHS_ms, 'VAS': HVHS_VAS, 'CP': HVHS_CP},
                             'HVLS': {'s': LVLS_s, 'ms': LVLS_ms, 'VAS': LVLS_VAS, 'CP': LVLS_CP},
                             'HVHS': {'s': LVHS_s, 'ms': LVHS_ms, 'VAS': LVHS_VAS, 'CP': LVHS_CP}
                             }  # this switch in label is due to inherent misalabelling in the json files where HV should be LV and vice-versa
    return participant_data_dict


def standardize_sequence(sent_sequence, scale, right, top, x_scaling, y_scaling):
    """
    Standardize the data
    Parameters
    ----------
    sequence : 2d List
        where 1st row is data from the x_axis and 2nd from the y_axis
    Returns
    -------
    sequence : 2d List
        now standardized
    """

    # STACKOVERFLOW: https://stackoverflow.com/questions/2612802/how-do-i-clone-a-list-so-that-it-doesnt-change-unexpectedly-after-assignment
    standardized_sequence = copy.deepcopy(sent_sequence)
    for i in range(0, len(standardized_sequence)):
        for j in range(0, len(standardized_sequence[0])):
            if i == 0 and standardized_sequence[i][j] != 0:
                standardized_sequence[i][j] = scale - ((standardized_sequence[i][j] - right) / x_scaling) * scale
            if i == 1 and standardized_sequence[i][j] != 0:
                standardized_sequence[i][j] = scale - ((standardized_sequence[i][j] - top) / y_scaling) * scale
    return standardized_sequence


def standardize_data(sequence, calibration_data, scale):
    """
    Standardize the sequence data
    Parameters
    ----------
    sequence : dict
        sequence to standardize
    calibration_data : str
        file that contains the calibration
        in our case this is: HPcompaqCalibration.mat
    scale : int
        DOUBT: for LVLS standardized on scale 100
    Returns
    -------
    sVAS : list
        standardized VAS
    sCP : list
        standardized CP
    """
    calibration_matrix = load_json(calibration_data)[0]['calibration']
    # print(calibration_matrix)
    left = calibration_matrix[0][0]
    right = calibration_matrix[0][1]
    # print('left: {} right: {}'.format(left, right))
    x_scaling = left - right

    top = calibration_matrix[1][0]
    bottom = calibration_matrix[1][2]
    # print('top: {} bottom: {}'.format(top, bottom))
    y_scaling = bottom - top

    # plt.scatter(calibration_matrix[0], calibration_matrix[1])
    # plt.show()

    sVAS = standardize_sequence(sequence['VAS'], scale, right, top, x_scaling, y_scaling)
    sCP = standardize_sequence(sequence['CP'], scale, right, top, x_scaling, y_scaling)

    standardized_sequence = {'CP': sCP, 'VAS': sVAS}
    return standardized_sequence


def process_participant_data(participant_data):
    participant_data_processed = {}
    for experiment_type in experiment_types:

        participant_data_processed[experiment_type] = {}
        ms_data = participant_data[experiment_type]['ms']
        CP_data = participant_data[experiment_type]['CP']
        VAS_data = participant_data[experiment_type]['VAS']

        # standardize the data
        data = {'CP': CP_data, 'VAS': VAS_data}
        standardized_data = standardize_data(data, 'data/json/HPcompaqCalibration.json', 100)

        sCP_data = standardized_data['CP']
        sVAS_data = standardized_data['VAS']

        predictions = []
        ratings = []

        # DOUBT: something is wrong with standardize
        # here change sCP to just CP to work with unstandardized data to see the difference
        for i in range(0, len(ms_data)):
            if ms_data[i] == 0 and i + 1 < len(ms_data):  # this will cut off final participant prediction
                # task is of prediction
                stimulus_now = ms_data[i + 1]
                confidence_of_pain_felt = sCP_data[0][i]
                pain_felt = sCP_data[1][i]
                trial_number = i
                predictions.append([stimulus_now, confidence_of_pain_felt, pain_felt, trial_number])
            elif ms_data[i] == -1:
                # task is of rating the felt pain
                stimulus_now = ms_data[i - 1]
                confidence_of_pain_felt = sVAS_data[0][i]
                pain_felt = sVAS_data[1][i]
                trial_number = i
                # here 0, 1, 2 -> stimulus_now, confidence_of_pain_felt, pain_felt
                ratings.append([stimulus_now, confidence_of_pain_felt, pain_felt, trial_number])
        # print('len of predictions: {}'.format(len(predictions)))
        if len(predictions) == 39:
            # adding a copy of the last values back to predictions
            last_prediction = predictions[-1][:3]
            dummy_trial_number = predictions[-1][3] + 1
            last_prediction.append(dummy_trial_number)
            predictions.append(last_prediction)
            # print('corrected len of predictions: {}'.format(len(predictions)))
        participant_data_processed[experiment_type]['CP'] = np.asarray(predictions)
        participant_data_processed[experiment_type]['VAS'] = np.asarray(ratings)
        participant_data_processed[experiment_type]['ms'] = np.asarray(ms_data)
    return participant_data_processed


# get all the participant_data
def pain_ratings_across_all_participants():
    all_participant_data = []
    parent_dir = 'data/json/'
    for file in sorted(os.listdir(parent_dir)):
        # don't want to take file HPcompaqCalibration.json
        if file[0] == 'p':
            # print(file)
            participant_data_dict = get_data(load_json(os.path.join(parent_dir, file)))
            all_participant_data.append(participant_data_dict)
    return all_participant_data


def converting_order(order_index):
    order = 4 * [None]
    j = 0
    for i in order_index:
        order[j] = experiment_types[i - 1]
        j = j + 1

    return order


def process_all_participants(all_participant_data):
    processed_data_all_participants_dict = {}
    for participant_data_dict in all_participant_data:
        processed_participant_data = process_participant_data(participant_data_dict)
        processed_data_all_participants_dict[participant_data_dict['participant_id']] = processed_participant_data
        processed_data_all_participants_dict[participant_data_dict['participant_id']]['Order'] = converting_order(
            participant_data_dict['Order'])

    return processed_data_all_participants_dict


# %%------------ Structure data and other functions
def cleaning_ms(ms):
    s_intensity = ms[ms != 0]
    s_intensity = s_intensity[s_intensity != -1]
    return s_intensity


def model_input_concat(processed_data_all_participants_dict):
    N = len(processed_data_all_participants_dict)  # storing all conditions
    tn = 80
    C = 1

    n_obs_VAS = int(tn / 2)
    n_obs_CP = 39

    SeqMat = np.zeros([4 * tn, N, C])
    # need to definition of pred index to convert to list
    Pred = np.zeros([n_obs_CP * 4, N, C])
    PredIndex = np.zeros([n_obs_CP * 4, N, C])
    PredConf = np.zeros([n_obs_CP * 4, N, C])

    Perc = np.zeros([n_obs_VAS * 4, N, C])
    PercIndex = np.zeros([n_obs_VAS * 4, N, C])
    PercConf = np.zeros([n_obs_VAS * 4, N, C])
    CondOrder = [0 for _ in range(N)]

    i = 0  # across participants
    for subj in processed_data_all_participants_dict:
        CondOrder[i] = processed_data_all_participants_dict[subj]['Order']
        VAS = []
        CP = []
        VAS_trial_number = []
        CP_trial_number = []
        seq = []
        VAS_conf = []
        CP_conf = []
        j = 0
        for condition in processed_data_all_participants_dict[subj]['Order']:
            ms = processed_data_all_participants_dict[subj][condition]['ms']
            seq = np.append(seq, cleaning_ms(ms))

            VAS = np.append(VAS, processed_data_all_participants_dict[subj][condition]['VAS'][:, 2])
            CP = np.append(CP, processed_data_all_participants_dict[subj][condition]['CP'][0:n_obs_CP, 2])

            VAS_trial_unadjusted = processed_data_all_participants_dict[subj][condition]['VAS'][:, 3]
            CP_trial_unadjusted = processed_data_all_participants_dict[subj][condition]['CP'][0:n_obs_CP, 3]

            VAS_trial_adjusted = (VAS_trial_unadjusted + 1) / 2 + j * tn
            CP_trial_adjusted = (CP_trial_unadjusted + 1) / 2 + j * tn

            VAS_trial_number = np.append(VAS_trial_number, VAS_trial_adjusted)
            CP_trial_number = np.append(CP_trial_number, CP_trial_adjusted)
            # shift by 1 as index in stan starts from 0

            VAS_conf = np.append(VAS_conf, processed_data_all_participants_dict[subj][condition]['VAS'][:, 1] / 100)
            CP_conf = np.append(CP_conf,
                                processed_data_all_participants_dict[subj][condition]['CP'][:n_obs_CP, 1] / 100)

            j += 1

        # indexing issue...
        CP = np.expand_dims(CP, axis=(1))
        CP_trial_number = np.expand_dims(CP_trial_number, axis=(1))
        CP_conf = np.expand_dims(CP_conf, axis=(1))

        VAS = np.expand_dims(VAS, axis=(1))
        VAS_trial_number = np.expand_dims(VAS_trial_number, axis=(1))
        VAS_conf = np.expand_dims(VAS_conf, axis=(1))

        Pred[:, i] = CP
        PredIndex[:, i] = CP_trial_number
        PredConf[:, i] = CP_conf

        Perc[:, i] = VAS
        PercIndex[:, i] = VAS_trial_number
        PercConf[:, i] = VAS_conf

        SeqMat[:, i] = np.expand_dims((seq - 1), axis=(
            1))  # minus one necessary so scale starts at 0, and can be easily multiplied for scaling factor

        i += 1

    PredIndex = PredIndex.astype(int)
    PredIndex = PredIndex.tolist()

    PercIndex = PercIndex.astype(int)
    PercIndex = PercIndex.tolist()

    model_data = {"N": N,
                  "Tn": tn * 4,
                  "N_obs_VAS": 4 * n_obs_VAS,
                  "N_obs_CP": 4 * n_obs_CP,
                  "SeqMat": SeqMat,
                  "Pred": Pred,
                  "PredIndex": PredIndex,
                  "PredConf": PredConf,
                  "Perc": Perc,
                  "PercIndex": PercIndex,
                  "PercConf": PercConf,
                  "C": C,
                  "CondOrder": CondOrder
                  }

    return model_data


def model_input(processed_data_all_participants_dict):
    N = len(processed_data_all_participants_dict)  # storing all conditions
    tn = 80
    C = 4

    n_obs_VAS = int(tn / 2)
    n_obs_CP = 39

    SeqMat = np.zeros([tn, N, C])
    # need to definition of pred index to convert to list
    Pred = np.zeros([n_obs_CP, N, C])
    PredIndex = np.zeros([n_obs_CP, N, C])
    PredConf = np.zeros([n_obs_CP, N, C])

    Perc = np.zeros([n_obs_VAS, N, C])
    PercIndex = np.zeros([n_obs_VAS, N, C])
    PercConf = np.zeros([n_obs_VAS, N, C])
    CondOrder = [0 for _ in range(N)]

    # need to add conf on scale from 0 to 1

    i = 0  # across participants
    for subj in processed_data_all_participants_dict:
        CondOrder[i] = processed_data_all_participants_dict[subj]['Order']
        j = 0  # across conditions

        # for condition in processed_data_all_participants_dict[subj]:
        for condition in experiment_types:
            if condition != 'Order':
                ms = processed_data_all_participants_dict[subj][condition]['ms']
                seq = cleaning_ms(ms)
                SeqMat[:, i,
                j] = seq - 1  # minus one necessary so scale starts at 0, and can be easily multiplied for scaling factor

                VAS = processed_data_all_participants_dict[subj][condition]['VAS'][:, 2]
                CP = processed_data_all_participants_dict[subj][condition]['CP'][0:n_obs_CP, 2]

                VAS_trial_number = processed_data_all_participants_dict[subj][condition]['VAS'][:, 3]
                CP_trial_number = processed_data_all_participants_dict[subj][condition]['CP'][0:n_obs_CP, 3]
                # VAS_trial_number = (processed_data_all_participants_dict[subj][condition]['VAS'][:,3]+1)/2
                # CP_trial_number = (processed_data_all_participants_dict[subj][condition]['CP'][0:n_obs_CP,3]+1)/2

                VAS_conf = processed_data_all_participants_dict[subj][condition]['VAS'][:, 1] / 100
                CP_conf = processed_data_all_participants_dict[subj][condition]['CP'][:n_obs_CP, 1] / 100

                # shift by 1 as index in stan starts from 0
                VAS_trial_number = (VAS_trial_number + 1) / 2

                # CP_trial_number = ((CP_trial_number+1)/2)+1
                CP_trial_number = ((CP_trial_number + 1) / 2)

                Pred[:, i, j] = CP
                PredIndex[:, i, j] = CP_trial_number.tolist()
                PredConf[:, i, j] = CP_conf

                Perc[:, i, j] = VAS
                PercIndex[:, i, j] = VAS_trial_number.tolist()
                PercConf[:, i, j] = VAS_conf

                j += 1

        i += 1

    PredIndex = PredIndex.astype(int)
    PredIndex = PredIndex.tolist()

    PercIndex = PercIndex.astype(int)
    PercIndex = PercIndex.tolist()

    model_data = {"N": N,
                  "C": C,
                  "Tn": tn,
                  "N_obs_VAS": n_obs_VAS,
                  "N_obs_CP": n_obs_CP,
                  "SeqMat": SeqMat,
                  "Pred": Pred,
                  "PredConf": PredConf,
                  "PredIndex": PredIndex,
                  "Perc": Perc,
                  "PercConf": PercConf,
                  "PercIndex": PercIndex,
                  "CondOrder": CondOrder}

    return model_data


# %%------------ Functions to organise data and transform functions - across conditoin and condition-wise

def line_fn(x, a, b):
    y = a * x + b
    return (y)


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def get_arrays_concat(model_data):
    stan_data = dict()
    N = model_data['N']
    NPerc = len(model_data['PercIndex'])
    NPred = len(model_data['PredIndex'])

    PredIndexArray = (np.array(model_data['PredIndex'])[:, :, 0]).T
    PercIndexArray = (np.array(model_data['PercIndex'])[:, :, 0]).T

    PredIndexArrayLong = np.zeros([N, model_data['Tn']]) - 1
    PercIndexArrayLong = np.zeros([N, model_data['Tn']]) - 1
    IndexArrayAll = np.zeros([N, model_data['Tn']]) - 1
    IndexMissingAll = np.zeros([N, model_data['Tn'] - (NPerc + NPred)]) - 1

    PercVals = model_data['Perc'][:, :, 0].T
    PredVals = model_data['Pred'][:, :, 0].T

    PercConf = model_data['PercConf'][:, :, 0].T
    PredConf = model_data['PredConf'][:, :, 0].T

    IntesAtPercVal = np.zeros([N, NPerc])
    IntesSeq = model_data['SeqMat'][:, :, 0].T

    for s in range(N):
        IntesAtPercVal[s, :] = IntesSeq[s, PercIndexArray[s, :] - 1]
        PredIndexArrayLong[s, PredIndexArray[s, :] - 1] = np.linspace(1, NPred, NPred)
        PercIndexArrayLong[s, PercIndexArray[s, :] - 1] = np.linspace(1, NPerc, NPerc)
        IndexArrayAll[s, PredIndexArray[s, :] - 1] = PredIndexArray[s, :]
        IndexArrayAll[s, PercIndexArray[s, :] - 1] = PercIndexArray[s, :]
        IndexMissingAll[s, :] = np.where(IndexArrayAll[s, :] == -1)[0] + 1

    stan_data['C'] = 1
    stan_data['N'] = N
    stan_data['Tn'] = model_data['SeqMat'].shape[0]
    stan_data['N_obs_CP'] = model_data['N_obs_CP']
    stan_data['N_obs_VAS'] = model_data['N_obs_VAS']

    stan_data['IntesSeq'] = IntesSeq

    stan_data['IntesAtPercVal'] = IntesAtPercVal

    stan_data['PercVals'] = PercVals
    stan_data['PercConf'] = PercConf

    stan_data['PredVals'] = PredVals
    stan_data['PredConf'] = PredConf

    stan_data['PercIndexArray'] = PercIndexArray
    stan_data['PredIndexArray'] = PredIndexArray

    stan_data['PercIndexArrayLong'] = PercIndexArrayLong
    stan_data['PredIndexArrayLong'] = PredIndexArrayLong

    stan_data['IndexMissingAll'] = IndexMissingAll.astype(int)

    TrialType = np.zeros([N, stan_data['Tn']]) - 1
    TrialTypeAllWMissing = np.zeros([N, stan_data['Tn']]) - 1
    PainValsAll = np.zeros([N, stan_data['Tn']]) - 1
    for s in range(N):
        TrialType[s, stan_data['PercIndexArray'][s, :] - 1] = 1
        TrialType[s, stan_data['PredIndexArray'][s, :] - 1] = 2
        TrialTypeAllWMissing[s, stan_data['PercIndexArray'][s, :] - 1] = 1
        TrialTypeAllWMissing[s, stan_data['PredIndexArray'][s, :] - 1] = 2
        TrialTypeAllWMissing[s, stan_data['IndexMissingAll'][s, :] - 1] = 2

        PainValsAll[s, stan_data['PercIndexArray'][s, :] - 1] = stan_data['PercVals'][s, :]
        PainValsAll[s, stan_data['PredIndexArray'][s, :] - 1] = stan_data['PredVals'][s, :]

    stan_data['TrialType'] = TrialType
    stan_data['TrialTypeAllWMissing'] = TrialTypeAllWMissing
    stan_data['PainValsAll'] = PainValsAll

    return stan_data


def get_arrays_cs(model_data_cs, experiment_types):
    stan_data_cs = dict.fromkeys(experiment_types, dict())
    N = model_data_cs['N']
    NPerc_cs = len(model_data_cs['PercIndex'])
    NPred_cs = len(model_data_cs['PredIndex'])
    # PredIndexArray_cs={};PercIndexArray_cs={}; PredIndexArrayLong_cs={}
    # PercIndexArrayLong_cs={}; IndexArrayAll_cs={}; IndexMissingAll_cs={}
    # IntesSeq_cs={}; PercVals_cs={}; PredVals_cs={}; IntesAtPercVal_cs={}
    # PercConf_cs={}; PredConf_cs={}
    for cond in experiment_types:
        stan_data_cs[cond] = dict.fromkeys(
            ['C', 'Cond', 'N', 'Tn', 'N_obs_CP', 'N_obs_VAS', 'IntesSeq', 'IntesAtPercVal', 'PercVals', 'PercConf',
             'PredVals', 'PredConf', 'PercIndexArray', 'PredIndexArray', 'PercIndexArrayLong', 'PredIndexArrayLong',
             'IndexMissingAll', 'TrialType', 'TrialTypeAllWMissing', 'PainValsAll', 'TransfIntesSeq',
             'TransfIntAtPercSeq'])
        PredIndexArray_cs = (np.zeros([N, NPred_cs]) - 1).astype(int)
        PercIndexArray_cs = (np.zeros([N, NPerc_cs]) - 1).astype(int)

        PredIndexArrayLong_cs = np.zeros([N, model_data_cs['Tn']])  # ignore
        PercIndexArrayLong_cs = np.zeros([N, model_data_cs['Tn']])  # ignore
        IndexArrayAll_cs = (np.zeros([N, model_data_cs['Tn']]) - 1).astype(int)
        IndexMissingAll_cs = (np.zeros([N, model_data_cs['Tn'] - (NPerc_cs + NPred_cs)]) - 1).astype(int)

        IntesSeq_cs = np.zeros([N, model_data_cs['Tn']]) - 1
        IntesAtPercVal_cs = np.zeros([N, NPerc_cs])
        PercVals_cs = np.zeros([N, NPerc_cs]) - 1
        PredVals_cs = np.zeros([N, NPred_cs]) - 1
        PercConf_cs = np.zeros([N, NPerc_cs]) - 1
        PredConf_cs = np.zeros([N, NPred_cs]) - 1

        for s in range(N):
            # c = model_data_cs['CondOrder'][s].index(cond)
            c = experiment_types.index(cond)
            IntesSeq_cs[s, :] = model_data_cs['SeqMat'][:, s, c]

            PredIndexArray_cs[s, :] = np.array(model_data_cs['PredIndex'])[:, s, c]
            PercIndexArray_cs[s, :] = np.array(model_data_cs['PercIndex'])[:, s, c]

            IntesAtPercVal_cs[s, :] = model_data_cs['SeqMat'][PercIndexArray_cs[s, :] - 1, s, c]

            PercVals_cs[s, :] = model_data_cs['Perc'][:, s, c]
            PredVals_cs[s, :] = model_data_cs['Pred'][:, s, c]
            PercConf_cs[s, :] = model_data_cs['PercConf'][:, s, c]
            PredConf_cs[s, :] = model_data_cs['PredConf'][:, s, c]

            IndexArrayAll_cs[s, PredIndexArray_cs[s, :] - 1] = PredIndexArray_cs[s, :]
            IndexArrayAll_cs[s, PercIndexArray_cs[s, :] - 1] = PercIndexArray_cs[s, :]
            IndexMissingAll_cs[s, :] = np.where(IndexArrayAll_cs[s, :] == -1)[0] + 1

            # PredIndexArrayLong[s,PredIndexArray[s,:]-1]=np.linspace(1,NPred,NPred)
            # PercIndexArrayLong[s,PercIndexArray[s,:]-1]=np.linspace(1,NPerc,NPerc)

        stan_data_cs[cond]['C'] = 1
        stan_data_cs[cond]['Cond'] = cond
        stan_data_cs[cond]['N'] = model_data_cs['N']
        stan_data_cs[cond]['Tn'] = model_data_cs['Tn']
        stan_data_cs[cond]['N_obs_CP'] = NPred_cs
        stan_data_cs[cond]['N_obs_VAS'] = NPerc_cs

        stan_data_cs[cond]['IntesSeq'] = IntesSeq_cs
        # stan_data_cs[cond]['TransfIntesSeq'] = TransfIntesSeq_cs[cond]

        stan_data_cs[cond]['IntesAtPercVal'] = IntesAtPercVal_cs
        # stan_data_cs[cond]['TransfIntAtPercSeq'] = TransfIntAtPercSeq_cs[cond]

        stan_data_cs[cond]['PercVals'] = PercVals_cs
        stan_data_cs[cond]['PercConf'] = PercConf_cs

        stan_data_cs[cond]['PredVals'] = PredVals_cs
        stan_data_cs[cond]['PredConf'] = PredConf_cs

        stan_data_cs[cond]['PercIndexArray'] = PercIndexArray_cs
        stan_data_cs[cond]['PredIndexArray'] = PredIndexArray_cs

        stan_data_cs[cond]['PercIndexArrayLong'] = PercIndexArrayLong_cs
        stan_data_cs[cond]['PredIndexArrayLong'] = PredIndexArrayLong_cs

        stan_data_cs[cond]['IndexMissingAll'] = IndexMissingAll_cs.astype(int)

        TrialType_cs = np.zeros([N, stan_data_cs[cond]['Tn']]) - 1
        TrialTypeAllWMissing_cs = np.zeros([N, stan_data_cs[cond]['Tn']]) - 1
        PainValsAll_cs = np.zeros([N, stan_data_cs[cond]['Tn']]) - 1

        for s in range(N):
            TrialType_cs[s, stan_data_cs[cond]['PercIndexArray'][s, :] - 1] = 1
            TrialType_cs[s, stan_data_cs[cond]['PredIndexArray'][s, :] - 1] = 2
            TrialTypeAllWMissing_cs[s, stan_data_cs[cond]['PercIndexArray'][s, :] - 1] = 1
            TrialTypeAllWMissing_cs[s, stan_data_cs[cond]['PredIndexArray'][s, :] - 1] = 2
            TrialTypeAllWMissing_cs[s, stan_data_cs[cond]['IndexMissingAll'][s, :] - 1] = 2

            PainValsAll_cs[s, stan_data_cs[cond]['PercIndexArray'][s, :] - 1] = stan_data_cs[cond]['PercVals'][s, :]
            PainValsAll_cs[s, stan_data_cs[cond]['PredIndexArray'][s, :] - 1] = stan_data_cs[cond]['PredVals'][s, :]

        stan_data_cs[cond]['TrialType'] = TrialType_cs
        stan_data_cs[cond]['TrialTypeAllWMissing'] = TrialTypeAllWMissing_cs
        stan_data_cs[cond]['PainValsAll'] = PainValsAll_cs
    return stan_data_cs


def get_linear_fit_tran(model_data, stan_data, stan_data_cs, IntesAtPercVal, PercVals, IntesSeq, save_fit_params,
                        experiment_types):
    N = model_data['N']
    NPerc = len(model_data['PercIndex'])
    # NPred = len(model_data['PredIndex'])

    TransfIntesSeq = np.zeros([N, IntesSeq.shape[1]])
    TransfIntAtPercSeq = np.zeros([N, NPerc])

    for s in range(N):
        xmin = np.min(IntesAtPercVal[s,])
        xmax = np.max(IntesAtPercVal[s,])
        inds_min = np.where(IntesAtPercVal[s,] == xmin)
        inds_max = np.where(IntesAtPercVal[s,] == xmax)
        ymin = np.average(PercVals[s, inds_min])
        ymax = np.average(PercVals[s, inds_max])
        a = (ymin - ymax) / (xmin - xmax)
        b = (xmin * ymax - xmax * ymin) / (xmin - xmax)

        save_fit_params['a'].loc[len(save_fit_params['a'])] = [pptIds[s], a, None, None]
        save_fit_params['b'].loc[len(save_fit_params['b'])] = [pptIds[s], b, None, None]

        xdata = IntesAtPercVal[s, :]
        ydata = PercVals[s, :]

        a_lr = linregress(xdata, ydata)[0]
        b_lr = linregress(xdata, ydata)[1]
        TransfIntesSeq[s, :] = a_lr * IntesSeq[s, :] + b_lr
        TransfIntAtPercSeq[s, :] = a_lr * IntesAtPercVal[s, :] + b_lr

        if np.sum(TransfIntesSeq[s, :] < 0):
            popt, pcov = curve_fit(line_fn, xdata, ydata, bounds=([-np.inf, 0.0], [np.inf, np.inf]))
            a_lr = popt[0]
            b_lr = popt[1]
            TransfIntesSeq[s, :] = a_lr * IntesSeq[s, :] + b_lr
            TransfIntAtPercSeq[s, :] = a_lr * IntesAtPercVal[s, :] + b_lr

        save_fit_params['a_lr'].loc[len(save_fit_params['a_lr'])] = [pptIds[s], a_lr, None, None]
        save_fit_params['b_lr'].loc[len(save_fit_params['b_lr'])] = [pptIds[s], b_lr, None, None]

    stan_data['TransfIntesSeq'] = TransfIntesSeq
    stan_data['TransfIntAtPercSeq'] = TransfIntAtPercSeq

    for cond in experiment_types:
        TransfIntesSeq_cs = np.zeros([N, stan_data_cs[cond]['IntesSeq'].shape[1]])
        TransfIntAtPercSeq_cs = np.zeros([N, stan_data_cs[cond]['PercVals'].shape[1]])

        for s in range(N):
            TransfIntesSeq_cs[s, :] = save_fit_params['a_lr'].iloc[s]['par_val'] * stan_data_cs[cond]['IntesSeq'][s,
                                                                                   :] + save_fit_params['b_lr'].iloc[s][
                                          'par_val']
            TransfIntAtPercSeq_cs[s, :] = save_fit_params['a_lr'].iloc[s]['par_val'] * stan_data_cs[cond][
                                                                                           'IntesAtPercVal'][s, :] + \
                                          save_fit_params['b_lr'].iloc[s]['par_val']

        stan_data_cs[cond]['TransfIntesSeq'] = TransfIntesSeq_cs
        stan_data_cs[cond]['TransfIntAtPercSeq'] = TransfIntAtPercSeq_cs

    return save_fit_params, stan_data, stan_data_cs


def save_data_pickle(stan_data, stan_data_cs, experiment_types):
    with open('stan_data/stan_lintran_data' + drop_str + '.pickle', 'wb') as handle:
        pickle.dump(stan_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for cond in experiment_types:
        with open('stan_data/stan_lintran_data_cs' + drop_str + '_' + cond + '.pickle', 'wb') as handle:
            pickle.dump(stan_data_cs[cond], handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%------------ Read in, structure, transform and save data for stan
all_participant_data = pain_ratings_across_all_participants()
processed_data_all_participants_dict = process_all_participants(all_participant_data)
outliers = ['718']
drop_out = True
drop_str = ''
if drop_out:
    drop_str = '_dropout'
    for k in outliers:
        processed_data_all_participants_dict.pop(k)
model_data = model_input_concat(processed_data_all_participants_dict)
model_data_cs = model_input(processed_data_all_participants_dict)
pptIds = [k for k in processed_data_all_participants_dict.keys()]

save_fit_params = {'a': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'b': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'a_lr': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'b_lr': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'A': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'K': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'B': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order']),
                   'M': pd.DataFrame(columns=['PID', 'par_val', 'cond', 'order'])}

stan_data = get_arrays_concat(model_data)
stan_data_cs = get_arrays_cs(model_data_cs, experiment_types)

save_fit_params, stan_data, stan_data_cs = get_linear_fit_tran(model_data, stan_data, stan_data_cs,
                                                               stan_data['IntesAtPercVal'], stan_data['PercVals'],
                                                               stan_data['IntesSeq'], save_fit_params, experiment_types)

pptIds_df = pd.DataFrame(pptIds, columns=['PID'])

# Save data
# save_data_pickle(stan_data,stan_data_cs,experiment_types)
# pptIds_df.to_csv('pptIds.csv',index=False)


# %%---- Plot linear fits across condtions - Supplement FigS1 ---------------- #########
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 17
fct = 3
N = stan_data['N']
fig, axes = plt.subplots(nrows=9, ncols=3, sharex=False, sharey=True, figsize=(8 * fct, 7.5 * fct))
# fig, axes = plt.subplots(nrows=7, ncols=4, sharex=False, sharey=True,figsize=(5*fct, 7*fct))
for s in range(N):
    a = save_fit_params['a_lr']['par_val'][s]
    b = save_fit_params['b_lr']['par_val'][s]

    xx = np.linspace(-5, 15, 200)
    yy = a * xx + b

    xdata = stan_data['IntesAtPercVal'][s, :]
    ydata = stan_data['PercVals'][s, :]

    unique_xx = np.unique(xdata)
    unique_yy = a * unique_xx + b

    if s < N + 1:
        ax = axes.flat[s]
        ax.plot(xx, yy, color='red', label='Linear fit')
        ax.scatter(xdata, ydata, s=10, label='Perception response', color='tab:blue')
        ax.plot(xx, np.zeros([xx.shape[0]]), color='gray', linestyle='dashed', linewidth=0.5)
        ax.scatter(unique_xx, unique_yy, color='black', s=50, label='Transformed input')
        ax.legend()
        # ax.ylabel('Pain rating',size=15)
        ax.set_xlabel('Stimulus intensity', size=25)
        if np.mod(s + 1, 4) == 1:
            ax.set_ylabel('Pain rating', size=25)
        ax.set_xlim([-7, 17])
        ax.set_ylim([-30, 130])
        ax.set_yticks(np.arange(-25, 150, 25))
        ax.set_xticks(np.arange(0, 16, 4))
        ax.set_title('Sub ' + pptIds[s] + ' a:' + str(round(a, 2)) + ', b:' + str(round(b, 2)) + '', size=20)

# plt.suptitle('Linear transformation of inputs - concatenated sequence.',y=0.99,horizontalalignment='center',fontsize=15)
plt.tight_layout()
plt.show()
# fig.delaxes(axes[6][3])
fig.savefig('plots/lintran_acrosscond' + drop_str + '.pdf', dpi=300)
fig.savefig('plots/FigS1.pdf', dpi=300)
# plt.close()

# %%--- Plot transformed input series for each condition
fct = 3
for cond in experiment_types:
    fig, axes = plt.subplots(nrows=7, ncols=4, sharex=False, sharey=True, figsize=(8 * fct, 5.5 * fct))
    N = stan_data_cs[cond]['N']
    for s in range(N):

        ax = axes.flat[s]
        if s < N + 1:
            ax.plot(stan_data_cs[cond]['TransfIntesSeq'][s,], color='red', label='nput')
            # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red')

            # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

            # ax.plot(xx,yy_sig, color='green',label='sigfit')
            # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
            # ax.scatter(xdata,ydata,s=10,label='data')
            # ax.scatter(unique_xx,unique_yy,color='black',s=50)
            # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
            ax.legend()
        # ax.ylabel('Pain rating',size=15)
        ax.set_xlabel('Trial', size=15)
        if np.mod(s + 1, 4) == 1:
            ax.set_ylabel('Input', size=15)
        # ax.set_xlim([-7,17])
        # ax.set_ylim([-30,130])
        # ax.set_yticks(np.arange(-25,150,25))
        ax.set_xticks(np.arange(0, 100, 20))
        ax.set_title('Sub ' + pptIds[s])
        # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
        #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

    plt.suptitle('Noxious input. ' + cond + ' condition.', y=0.99, horizontalalignment='center', fontsize=15)
    fig.tight_layout()
    # fig.savefig('plots/traninput'+drop_str+'_'+cond+'.pdf',dpi=300)
    plt.close()

# %%--- Plot transformed input series for concatenated sequence - Figure XXX
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
fct = 3
fig, axes = plt.subplots(nrows=9, ncols=3, sharex=False, sharey=True, figsize=(8 * fct, 7.5 * fct))
N = stan_data['N']
for s in range(N):
    # processed_data_all_participants_dict[pptIds[s]][cond]['ms'][(processed_data_all_participants_dict[pptIds[s]][cond]['VAS'][:,3]).astype(int)]
    # model_data_cs['SeqMat'][:,1,c]

    ax = axes.flat[s]
    if s < N + 1:
        ax.plot(stan_data['TransfIntesSeq'][s,], color='red')
        ax.vlines(x=[80, 160, 240, 320], ymin=0, ymax=100, colors='purple')
        # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red') 

        # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

        # ax.plot(xx,yy_sig, color='green',label='sigfit')
        # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
        # ax.scatter(xdata,ydata,s=10,label='data')
        # ax.scatter(unique_xx,unique_yy,color='black',s=50)
        # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
        # ax.legend()
    # ax.ylabel('Pain rating',size=15)
    ax.set_xlabel('Trial', size=25)
    if np.mod(s + 1, 3) == 1:
        ax.set_ylabel('Input', size=25)
    # ax.set_xlim([-7,17])
    # ax.set_ylim([-30,130])
    ax.set_yticks(np.arange(0, 150, 50))
    ax.set_title('Sub ' + pptIds[s], size=25)
    ax.set_xticks(np.arange(0, 340, 80))
    # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
    #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

# plt.suptitle('Noxious input',y=0.99,horizontalalignment='center',fontsize=25)
fig.tight_layout()
fig.savefig('plots/traninput_concat' + drop_str + '.pdf', dpi=300)
plt.close()

# %%--- Plot confidence rating for each condition
fct = 3
# for c in range(len(experiment_types)):
for cond in experiment_types:
    fig, axes = plt.subplots(nrows=7, ncols=4, sharex=False, sharey=True, figsize=(8 * fct, 5.5 * fct))
    N = stan_data_cs[cond]['N']
    for s in range(N):

        ax = axes.flat[s]
        if s < N + 1:
            # ax.plot(stan_data_cs[cond]['TransfIntesSeq'][s,], color='red')
            ax.plot(stan_data_cs[cond]['PercIndexArray'][s,], stan_data_cs[cond]['PercConf'][s,], color='red',
                    label='perception')
            ax.plot(stan_data_cs[cond]['PredIndexArray'][s,], stan_data_cs[cond]['PredConf'][s,], color='green',
                    label='prediction')
            # ax.vlines(x = [80,160,240,320], ymin = -0.25, ymax = 1, colors = 'purple')

            # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red')

            # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

            # ax.plot(xx,yy_sig, color='green',label='sigfit')
            # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
            # ax.scatter(xdata,ydata,s=10,label='data')
            # ax.scatter(unique_xx,unique_yy,color='black',s=50)
            # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
            ax.legend()
        # ax.ylabel('Pain rating',size=15)
        ax.set_xlabel('Trial', size=15)
        if np.mod(s + 1, 4) == 1:
            ax.set_ylabel('Confidence', size=15)
        # ax.set_xlim([-7,17])
        # ax.set_ylim([-30,130])
        # ax.set_yticks(np.arange(-25,150,25))
        ax.set_xticks(np.arange(0, 100, 20))
        ax.set_ylim(-0.25, 1.1)
        ax.set_title('Sub ' + pptIds[s])
        # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
        #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

    plt.suptitle('Confidence rating ' + cond + ' condition.', y=0.99, horizontalalignment='center', fontsize=15)
    fig.tight_layout()
    # fig.savefig('plots/confratings'+drop_str+'_'+cond+'.pdf',dpi=300)
    plt.close()

# %%--- Plot confidence rating for concatenated sequence - Figure S3
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
fct = 3
fig, axes = plt.subplots(nrows=9, ncols=3, sharex=False, sharey=True, figsize=(8 * fct, 7.5 * fct))
N = stan_data['N']
for s in range(N):

    ax = axes.flat[s]
    if s < N + 1:
        ax.plot(stan_data['PercIndexArray'][s,], stan_data['PercConf'][s,], color='#D81B60', label='perception')
        ax.plot(stan_data['PredIndexArray'][s,], stan_data['PredConf'][s,], color='#FFC107', label='prediction')
        ax.vlines(x=[80, 160, 240, 320], ymin=-0.25, ymax=1, colors='purple')
        # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red')

        # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

        # ax.plot(xx,yy_sig, color='green',label='sigfit')
        # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
        # ax.scatter(xdata,ydata,s=10,label='data')
        # ax.scatter(unique_xx,unique_yy,color='black',s=50)
        # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
        ax.legend()
    # ax.ylabel('Pain rating',size=15)
    ax.set_xlabel('Trial', size=25)
    if np.mod(s + 1, 3) == 1:
        ax.set_ylabel('Confidence', size=25)
    # ax.set_xlim([-7,17])
    # ax.set_ylim([-30,130])
    ax.set_yticks(np.arange(0, 1.5, 0.5))
    ax.set_title('Sub ' + pptIds[s], size=25)
    ax.set_ylim(-0.25, 1.1)
    ax.set_xticks(np.arange(0, 340, 80))
    # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
    #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

# plt.suptitle('Confidence rating',y=0.99,horizontalalignment='center',fontsize=25)
fig.tight_layout()
fig.savefig('plots/confratings_concat' + drop_str + '.pdf', dpi=300)
fig.savefig('plots/FigS3_v2.pdf', dpi=300)
plt.close()

# %%--- Plot responses for each condition
fct = 3
# for c in range(len(experiment_types)):
for cond in experiment_types:
    fig, axes = plt.subplots(nrows=7, ncols=4, sharex=False, sharey=True, figsize=(8 * fct, 5.5 * fct))
    N = stan_data_cs[cond]['N']
    for s in range(N):
        # processed_data_all_participants_dict[pptIds[s]][cond]['ms'][(processed_data_all_participants_dict[pptIds[s]][cond]['VAS'][:,3]).astype(int)]
        # model_data_cs['SeqMat'][:,1,c]

        ax = axes.flat[s]
        if s < N + 1:
            # ax.plot(stan_data_cs[cond]['TransfIntesSeq'][s,], color='red')
            ax.plot(stan_data_cs[cond]['PercIndexArray'][s,], stan_data_cs[cond]['PercVals'][s,], color='red',
                    label='perception')
            ax.plot(stan_data_cs[cond]['PredIndexArray'][s,], stan_data_cs[cond]['PredVals'][s,], color='green',
                    label='prediction')
            ax.plot(stan_data_cs[cond]['TransfIntesSeq'][s,], color='black', label='input', linestyle='dotted')
            # ax.vlines(x = [80,160,240,320], ymin = -0.25, ymax = 1, colors = 'purple')

            # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red')

            # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

            # ax.plot(xx,yy_sig, color='green',label='sigfit')
            # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
            # ax.scatter(xdata,ydata,s=10,label='data')
            # ax.scatter(unique_xx,unique_yy,color='black',s=50)
            # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
            ax.legend()
        # ax.ylabel('Pain rating',size=15)
        ax.set_xlabel('Trial', size=15)
        if np.mod(s + 1, 4) == 1:
            ax.set_ylabel('Responser', size=15)
        # ax.set_xlim([-7,17])
        # ax.set_ylim([-30,130])
        # ax.set_yticks(np.arange(-25,150,25))
        ax.set_xticks(np.arange(0, 100, 20))
        ax.set_ylim(-10, 110)
        ax.set_title('Sub ' + pptIds[s])
        # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
        #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

    plt.suptitle('Responses ' + cond + ' condition.', y=0.99, horizontalalignment='center', fontsize=15)
    fig.tight_layout()
    # fig.savefig('plots/responses'+drop_str+'_'+cond+'.pdf',dpi=300)
    plt.close()

# %%--- Plot responses for concatenated sequence --- FigS2
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
fct = 3
fig, axes = plt.subplots(nrows=9, ncols=3, sharex=False, sharey=True, figsize=(8 * fct, 7.5 * fct))
N = stan_data['N']
for s in range(N):
    # processed_data_all_participants_dict[pptIds[s]][cond]['ms'][(processed_data_all_participants_dict[pptIds[s]][cond]['VAS'][:,3]).astype(int)]
    # model_data_cs['SeqMat'][:,1,c]

    ax = axes.flat[s]
    if s < N + 1:
        ax.plot(stan_data['PercIndexArray'][s,], stan_data['PercVals'][s,], color='#D81B60', label='perception')
        ax.plot(stan_data['PredIndexArray'][s,], stan_data['PredVals'][s,], color='#FFC107', label='prediction')
        ax.plot(stan_data['TransfIntesSeq'][s,], color='#1E88E5', label='input', linestyle='dotted')
        ax.vlines(x=[80, 160, 240, 320], ymin=-10, ymax=110, colors='purple')
        # ax.plot(model_data_cs['SeqMat'][:,s,c], color='red')

        # ax.plot(processed_data_all_participants_dict[pptIds[s]][cond]['ms'], color='red',label='linfit')

        # ax.plot(xx,yy_sig, color='green',label='sigfit')
        # ax.plot(xx,np.zeros([xx.shape[0]]),color='gray',linestyle='dashed',linewidth=0.5)
        # ax.scatter(xdata,ydata,s=10,label='data')
        # ax.scatter(unique_xx,unique_yy,color='black',s=50)
        # ax.scatter(unique_xx,unique_yy_sig,color='green',s=25)
        ax.legend()
    # ax.ylabel('Pain rating',size=15)
    ax.set_xlabel('Trial', size=25)
    if np.mod(s + 1, 3) == 1:
        ax.set_ylabel('Response', size=25)
    # ax.set_xlim([-7,17])
    # ax.set_ylim([-30,130])
    ax.set_yticks(np.arange(0, 150, 50))
    ax.set_title('Sub ' + pptIds[s], size=25)
    ax.set_ylim(-10, 110)
    ax.set_xticks(np.arange(0, 340, 80))
    # ax.set_title('Sub '+pptIds[s]+', a:'+str(round(a,2))+', b:'+str(round(b,2))+
    #              ', A:'+str(round(A,2))+', \nK:'+str(round(K,2))+', B:'+str(round(B,2))+', M:'+str(round(M,2)))

# plt.suptitle('Responses ',y=0.99,horizontalalignment='center',fontsize=15)
fig.tight_layout()
fig.savefig('plots/responses_concat' + drop_str + '.pdf', dpi=300)
fig.savefig('plots/FigS2.pdf', dpi=300)
plt.close()

# %%--- Export coefficents from linear transformation
save_fit_params['a_lr'].to_csv('a_coeffs.csv', index=False)
save_fit_params['b_lr'].to_csv('b_coeffs.csv', index=False)
