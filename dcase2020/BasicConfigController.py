import os
import json


class BasicConfigController(object):
    def __init__(self, path='../workspace/basic_cfg.json'):
        super(BasicConfigController, self).__init__()
        self.dummy = {"threshold": 61.0}
        self.path = path
        self.values = self.__read_json()

    def __read_json(self):
        if os.path.isfile(self.path):
            values = json.load(open(self.path))
        else:
            self.__create_json()
            values = self.dummy
        return values

    def __create_json(self, path=None, content=None):
        path = path if content is not None else self.path
        fp = open(path, "w")
        json.dump(content if content is not None else self.dummy, fp, indent=4)
        fp.close()

    def __compare_values(self, values):
        if self.values == values:
            return False
        else:
            return True

    def __update_values(self, values):
        self.values = values

    def is_updated(self):
        values = self.__read_json()
        flag = self.__compare_values(values)
        if flag:
            self.__update_values(values)
        return flag

    def values(self):
        return self.values

    def get_value(self, key):
        return self.values[key]