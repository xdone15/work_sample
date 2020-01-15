import pickle
import pywt
import time
import numpy as np
from scipy import signal
from concurrent import futures
from collections import Iterable
from functools import wraps


# util decorators
# timer deco
def func_timer(switch=False):
    if switch:
        def foo_timer(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = f(*args, **kwargs)
                print('running time for function {0} is {1:.3f}'.format(f.__name__, time.time()-start_time))
                return result
            return wrapper
        return foo_timer
    else:
        def foo(f):
            return f
        return foo

def func_dump(switch=False):
    if switch:
        def foo_dump(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                result = f(*args, **kwargs)
                with open('{0}'.format(f.__name__)+'.pickle', 'wb') as file:
                    pickle.dump(result, file)
                return result
            return wrapper
        return foo_dump
    else:
        def foo(f):
            return f
        return foo


def limited_memoize(f):
    # Use with caution: this is simplified memoize decorator with limited argument compatibility
    # decorated function MUST have save_id as known argument to activate and save by that key
    memo = {}
    @wraps(f)
    def wrapper(*args, **kwargs):
        save_id = kwargs.get('save_id', None)
        if save_id:
            if save_id not in memo:
                memo[save_id] = f(*args, **kwargs)
            return memo[save_id]
        else:
            return f(*args, **kwargs)
    return wrapper


def data_loader(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def mp_apply(func, input_iter, local=False, *args, **kwargs):
    """
    Simplified function for sending multiprocessing task on function
    *limited functionality
    :param func: function object
    :param input_iter: iterable input that compatible with func
    :param local: bool value decide if run func on local
    :param args: other positional argument
    :param kwargs:
    :return: list of result from multiprocss
    """
    assert isinstance(input_iter, Iterable), 'input for the function need to be iterable'
    if local:
        result = []
        for input_value in input_iter:
            result.append(func(input_value, *args, **kwargs))
        return result
    else:
        with futures.ProcessPoolExecutor() as e:
            result = e.map(lambda x: func(x, *args, **kwargs), input_iter)

        return result


# TODO : generalize functionality
class TripMatch(object):
    """
    For mobile-trip/odb2-trip data -
    """
    def __init__(self, mobile_data, obd_data, **kwargs):
        self.mobile_data = mobile_data
        self.obd_data = obd_data
        self.kwargs = kwargs

    def __call__(self, call_func):
        # for conveniences to generate all results mentioned in report
        # todo : extend to more functions
        supported_call = {
            'x_cor': self.run_xcor
        }

        if call_func not in supported_call.keys():
            raise ValueError('{0} does not support'.format(call_func))
        else:
            return supported_call[call_func]()

    # todo : add pre-processing if necessary
    def prep_data(self):
        self.mobile_data.sort(key=lambda x: x['created_at'])
        self.obd_data.sort(key=lambda x: x['created_at'])

    @staticmethod
    def standardize(array):
        n = array.ndim
        if n == 1:
            return (array - array.mean())/array.std()
        elif n == 2:
            return (array - array.mean(1, keepdims=True))/array.std(1, keepdims=True)
        else:
            raise ValueError('only support 1d or 2d array')

    # todo : improve efficiency | threading/multi-processing
    @func_dump(True)
    def run_xcor(self):
        # run cross_correlation analysis

        excluded_trip = {'mobile': [], 'obdII': []}

        matched_trip = []
        obd_trip_start_ix = 0
        min_match_length = self.kwargs.get('min_match_thresh', 300)  #
        min_length = self.kwargs.get('min_length', 120)
        error_thresh = self.kwargs.get('min_error_thresh', 10.)
        length_thresh = min_match_length/min_length

        for m_trip_ix, m_trip in enumerate(self.mobile_data):

            min_error = float('inf')
            match_candidate, match_candidate_info, match_candidate_ix = None, None, None

            if m_trip['data']['speed'].mean() == 0.0:
                excluded_trip['mobile'] += [m_trip['trip_id']]
                continue

            m_trip_array = m_trip['data']['speed'].values
            m_trip_length = len(m_trip_array)
            cutout = m_trip_length - m_trip_length % min_length
            number_of_split = m_trip_length//min_length  # number of split
            m_trip_split = np.array_split(m_trip_array[:cutout], number_of_split)
            m_trip_split = self.standardize(np.array(m_trip_split))

            for o_trip_ix in np.arange(obd_trip_start_ix, len(self.obd_data)):

                o_trip = self.obd_data[o_trip_ix]

                if o_trip['trip_id'] in excluded_trip['obdII']:
                    continue

                elif o_trip['data']['speed'].mean() == 0.0:
                    excluded_trip['obdII'] += [o_trip['trip_id']]
                    continue

                o_trip_array = o_trip['data']['speed'].values
                o_trip_matrix = self.match_with_correlation_prep(o_trip_array, save_id=o_trip['trip_id'])
                match_info = self.match_with_correlation(m_trip_split, o_trip_matrix, short_version=True)

                if match_info['error'] < min_error:
                    min_error = match_info['error']
                    match_candidate = o_trip
                    match_candidate_info = match_info
                    match_candidate_ix = o_trip_ix

            if min_error < error_thresh and match_candidate_info['match_length'] >= length_thresh:
                obd_trip_start_ix = match_candidate_ix + 1
                matched_trip.append({'id': (m_trip['trip_id'], match_candidate['trip_id']),
                                     'index': (m_trip_ix, match_candidate_ix),
                                     'info':  match_candidate_info})

        return matched_trip, excluded_trip

    @limited_memoize
    def match_with_correlation_prep(self, cmp_path, save_id):
        _ = save_id  # save_id no use in function itself - null var _ as convention
        min_trip_length = self.kwargs.get('min_length', 120)
        cmp_matrix = np.array([cmp_path[i:i+min_trip_length] for i in range(len(cmp_path)-min_trip_length)])
        cmp_matrix = self.standardize(cmp_matrix)
        return cmp_matrix/cmp_matrix.shape[1]

    def match_with_correlation(self, ref_path_split, cmp_matrix, **kwargs):
        """
        :param ref_path_split: reference path for matching
        :param cmp_matrix: compare path for
        :return: match info
        """
        short_version = kwargs.get('short_version', False)
        # note: np.vectorize does NOT improve performance
        call_calc = np.vectorize(self.corrcoef_calc, excluded=['cmp_matrix', 'short_version'], signature='(n)->(),()')
        result = call_calc(ref_path_split, cmp_matrix=cmp_matrix, short_version=short_version)
        match_info = self.calc_corr_warping_error(result)
        return match_info

    def calc_corr_warping_error(self, result):
        """
        :param result: (array of index positions, array of correlation)
        :return: dictionary of match information
        """
        # calculate "warping" score based correlation
        min_trip_length = self.kwargs.get('min_length', 120)
        thresh = self.kwargs.get('corr_thresh', 0.9)
        high_cor_position, match_cor = result
        whr = match_cor >= thresh
        match_position = high_cor_position[whr]
        match_length = len(match_position)

        if len(match_position) <= 1.:
            return {'error': float('inf'), 'match_length': match_length, 'match_position': None}
        else:
            error = np.abs(np.diff(match_position)-min_trip_length).mean()
            return {'error': error,
                    'match_length': match_length,
                    'match_position': (np.where(whr)[0][0], match_position[0])}

    @staticmethod
    def corrcoef_calc(ref_vector, cmp_matrix, short_version=False):
        """
        vectorized correlation calculation
        :param ref_vector: reference vector
        :param cmp_matrix: vectors for compare
        :param short_version: bool if use short version (inputs are properly normalized )
        :return: index of highest correlation and value
        """
        if short_version:
            corr_coef_vector = np.sum(cmp_matrix*ref_vector, 1)

        else:
            cmp_matrix = cmp_matrix - cmp_matrix.mean(1, keepdims=True)
            ref_vector = ref_vector - ref_vector.mean()
            corr_coef_vector = (cmp_matrix*ref_vector).sum(1)/np.sqrt((cmp_matrix**2).sum(1)*(ref_vector**2).sum())

        try:
            ix = np.nanargmax(corr_coef_vector)
        except ValueError:
            return 0., 0.

        return ix, corr_coef_vector[ix]

    # -------- decomposition features
    @staticmethod
    def fft_feature(input_series, **kwargs):

        time_step = kwargs.get('time_step', 1)
        length = input_series.size
        k = length//2  # frequency parameter, cannot be more than half of the signal
        fft_value = np.fft.fft(input_series)
        fft_rfreq = np.fft.rfftfreq(length, d=time_step)[:-1]
        amplitude = 2.0*np.abs(fft_value)[:k]/length
        angle = np.angle(fft_value)
        return fft_rfreq, amplitude, angle

    @staticmethod
    def discrete_wavelet_feature(input_series, wavelet_obj='db1', **kwargs):
        low_pass, high_pass = pywt.cwt(input_series, wavelet_obj, **kwargs)
        return low_pass, high_pass

    @staticmethod
    def welch_feature(input_series, **kwargs):
        freq, power_spec = signal.welch(input_series, **kwargs)
        return freq, power_spec


if __name__ == '__main__':
    mobile_data_load = data_loader('./mobile.pickle')
    obd_data_load = data_loader('./obd2.pickle')
    run_class = TripMatch(mobile_data_load, obd_data_load)
    results = run_class('x_cor')

    print('done')