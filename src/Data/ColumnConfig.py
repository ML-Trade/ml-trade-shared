import json
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from enum import Enum

class NormFunction(Enum):
    STD = "std"
    MA_STD = "ma_std"
    MINMAX = "minmax"
    TIME = "time"


class ColumnConfig:
    """
    The configuration associated with preprocessing a dataframe for each column.
    This is to be passed into a Data Preprocessor such as TSDataPreprocessor.

    JSON Format:
    ```
    [
        {
            name: "SOME_NAME",
            norm_function: "std | ma_std | minmax | time",
            
            period: 69,
            class_meanings: ["BUY", "SELL"],
            is_target: false,
            mean: 10.5,
            std: 10.5
        },
        ...
    ]
    ```
    The order of the array is important; this is the order of the columns in the df / numpy sequence array
    
    "period" is optional and is used for ma_std (moving average then pct_change then standardise)
    "class_meanings" is optional. It is only relevant when is_target is true
    "is_target" is optional. It is only relevant when it is the target column
    "mean" and "std" is optional. They are only added by a data preprocessor when using std or ma_std norm_function
    
    """
    def __init__(self, columns: List[str], *, target_column: str, class_meanings: List[str]) -> None:
        self.config: OrderedDict = OrderedDict()
        for col in columns:
            self.config[col] = {}
            if col == target_column:
                self.config[col]["class_meanings"] = class_meanings
                self.config[col]["is_target"] = True
        self.target_column = target_column
        self.default_function = NormFunction.STD
        self.default_function_args: Dict[str, Any] = {}


    def set_target(self, col_name: str, class_meanings: Optional[List[str]] = None):
        class_meanings = class_meanings or self.config[self.target_column]["class_meanings"]
        del self.config[self.target_column]["class_meanings"]
        del self.config[self.target_column]["is_target"]
        self.target_column = col_name
        self.config[col_name]["class_meanings"] = class_meanings
        self.config[col_name]["is_target"] = True

    def set_default_norm_function(self, norm_function: NormFunction, args: dict = {}):
        self.default_function = norm_function
        self.default_function_args = args

    def to_dict(self) -> OrderedDict:
        config: OrderedDict[str, Any] = OrderedDict()
        for col_name, value in self.config.items():
            config[col_name] = {}
            if "norm_function" not in value:
                config[col_name] = {"norm_function": self.default_function, **self.default_function_args, **value}
            else:
                config[col_name] = value
        return config
                
    def set_norm_function(self, name: str, norm_function: NormFunction, args: dict = {}):
        self.config[name]["norm_function"] = norm_function
        self.config[name] = {**args, **self.config[name]}

    def add_args(self, name: str, args: dict = {}):
        self.config[name] = {**args, **self.config[name]}

    def to_json(self) -> str:
        config = self.to_dict()
        # Saved as an array to maintain ordering when serialising and deserialising JSON
        config_arr = []
        for col_name, value in config.items():
            config_value = value
            config_value["name"] = col_name
            config_value["norm_function"] = config_value["norm_function"].value
            config_arr.append(config_value)
        return json.dumps(config_arr)

    @staticmethod
    def from_json(json_string: str):
        config_arr: List[dict] = json.loads(json_string)
        colname_arr = []
        class_meanings = []
        target_column = ""
        for val in config_arr:
            colname_arr.append(val["name"])
            if "is_target" in val and val["is_target"] == True:
                class_meanings = val["class_meanings"]
                target_column = val["name"]
        col_config = ColumnConfig(columns=colname_arr, target_column=target_column, class_meanings=class_meanings)
        for val in config_arr:
            val_args = val.copy()
            del val_args["name"]
            col_config.set_norm_function(val["name"], NormFunction(val["norm_function"]), val_args)
        return col_config