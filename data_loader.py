import json
import gzip
import os
import logging
import pickle
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mobile_trip_file = "mobile_trips.json.gz"
obd2_trip_file = "obd2_trips.json.gz"


#  load and pickle data #
def file_loader(file_name):
    """
    load .gz file and parse json
    :param file_name: target file directory
    :return: None or parsed dictionary
    """
    if not os.path.exists(file_name):
        logger.warning('file does not exist')
        return None
    else:
        with gzip.open(file_name) as f:
            loaded_json = f.read()

    return json.loads(loaded_json)


# convert raw data to dictionary of data frames
def process_data(input_data, save_to = None):
    """
    :param input_data: data parsed from json
    :rtype input_data: list of list
    :return: list
    """
    def func_to_dict(trip_data):
        df = pd.DataFrame(trip_data, columns = ['timestamp', 'speed', 'accuracy'])
        return {'trip_id': trip_data[0].get('trip_id', None),
                'created_at': trip_data[0].get('created_at',None),
                'data': df}

    result = [func_to_dict(trip_data) for trip_data in input_data]

    if save_to:
        with open('./'+save_to+'.pickle', 'wb') as pkl_f:
            pickle.dump(result, pkl_f)

    return result


if __name__ == "__main__":
    mobile_data = file_loader(mobile_trip_file)
    obd2_data = file_loader(obd2_trip_file)

    mobile = process_data(mobile_data, None)
    obd2 = process_data(obd2_data, None)

    print('done')