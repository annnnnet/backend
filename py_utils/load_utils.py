import json
import sys, os


def save_json(json_object, data_file):
    with open(data_file, 'w') as df:
        json_object_string = json.dumps(json_object)
        json.dump(json_object_string, df)
