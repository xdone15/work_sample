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
    # decorated function MUST have id as known argument to activate
    memo = {}
    @wraps(f)
    def wrapper(*args,**kwargs):
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

    # todo : improve efficiency | threading/muliprocessing
    @func_dump(False)
    def run_xcor(self):
        # run cross_correlation analysis

        excluded_trip = []
        matched_trip = []
        obd_trip_start_ix = 0
        min_match_length = self.kwargs.get('min_match_thresh', 300)  #
        min_length = self.kwargs.get('min_length', 120)
        error_thresh = self.kwargs.get('min_error_thresh', 10.)
        length_thresh = min_match_length/min_length

        for m_trip_ix in np.arange(len(self.mobile_data)):

            min_error = float('inf')
            match_candidate, match_candidate_info, match_candidate_ix = None, None, None
            m_trip = self.mobile_data[m_trip_ix]

            if m_trip['data']['speed'].mean() == 0.0:
                excluded_trip.append(('mobile', m_trip['trip_id']))
                continue

            for o_trip_ix in np.arange(obd_trip_start_ix, len(self.obd_data)):
                o_trip = self.obd_data[o_trip_ix]

                if o_trip['data']['speed'].mean() == 0.0:
                    excluded_trip.append(('obdII', o_trip['trip_id']))
                    continue

                m_trip_array = m_trip['data']['speed'].values
                o_trip_array = o_trip['data']['speed'].values
                match_info = self.match_with_correlation(m_trip_array, o_trip_array)

                if match_info['error'] < min_error:
                    min_error = match_info['error']
                    match_candidate = o_trip
                    match_candidate_info = match_info
                    match_candidate_ix = o_trip_ix

            if min_error < error_thresh and match_candidate_info['match_length'] >= length_thresh:
                matched_trip.append({'id': (m_trip['trip_id'], match_candidate['trip_id']),
                                     'index': (m_trip_ix, match_candidate_ix),
                                     'info':  match_candidate_info})

        return matched_trip, excluded_trip

    def match_benchmark(self, ref_path, cmp_path_pd):
        # benchmark test :
        min_trip_length = self.kwargs.get('min_length', 120)
        ref_path_split = np.array_split(ref_path, np.arange(0, len(ref_path), min_trip_length)[1:])


        match_array = cmp_path_pd.rolling(min_trip_length).apply(lambda x : np.correlate)


    def match_with_correlation(self, ref_path, cmp_path):
        """

        :param ref_path: reference path for matching
        :param cmp_path: compare path for
        :return: match info
        """

        min_trip_length = self.kwargs.get('min_length', 120)
        cmp_matrix = np.array([cmp_path[i:i+min_trip_length] for i in range(len(cmp_path)-min_trip_length)])
        cmp_matrix = cmp_matrix - cmp_matrix.mean(1, keepdims=True)

        length = len(ref_path)
        cutout = length - length % min_trip_length
        m = length // min_trip_length

        ref_path_split = np.array_split(ref_path[:cutout], m)
        call_calc = np.vectorize(self.corrcoef_vector, excluded=['cmp_matrix', 'de_mean'], signature='(n)->(),()')
        result = call_calc(ref_path_split, cmp_matrix=cmp_matrix, de_mean=False)
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
    def standardize(array):
        return (array - array.mean())/array.std()

    @staticmethod
    def corrcoef_vector(ref_vector, cmp_matrix, de_mean=True):

        if de_mean:
            cmp_matrix = cmp_matrix - cmp_matrix.mean(1, keepdims=True)

        ref_vector = ref_vector - ref_vector.mean()
        # print(np.sqrt((cmp_matrix ** 2).sum(1) * (ref_vector ** 2).sum()))
        corr_coef_vector = (cmp_matrix*ref_vector).sum(1)/np.sqrt((cmp_matrix**2).sum(1)*(ref_vector**2).sum())

        try:
            ix = np.nanargmax(corr_coef_vector)
        except ValueError:
            return 0., 0.

        return ix, corr_coef_vector[ix]

    # -------- decomposition features
    @staticmethod
    def fft_feature(input_series, **kwargs):

        timestep = kwargs.get('timestep', 1)
        length = input_series.size
        k = length//2  # frequency parameter, cannot be more than half of the signal
        fft_value = np.fft.fft(input_series)
        fft_rfreq = np.fft.rfftfreq(length, d=timestep)[:-1]
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

    i0 = mobile_data_load[0]['data']['speed'].values
    for d in obd_data_load[:1]:
        res = run_class.match_benchmark(i0, d['data']['speed'].values)
    print(res)

    print('done')