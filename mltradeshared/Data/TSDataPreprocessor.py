from collections import deque
from datetime import datetime
import os
from re import M
from typing import Deque, Dict, Callable, List, Optional, Union
import pandas as pd
import numpy as np
import hashlib
import glob
from dataclasses import dataclass

from .ColumnConfig import ColumnConfig, NormFunction
from .Dataset import Dataset, DatasetMetadata


##  TODO: move this to utils
def minmax_norm(array: Union[list, np.ndarray], minimum: Optional[float] = None, maximum: Optional[float] = None) -> pd.Series:
    maximum = maximum or np.max(array)
    minimum = minimum or np.min(array)
    np_array = np.array(array)
    return pd.Series((np_array - minimum) / (maximum - minimum))

def standardise(arr: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None):
    mean = mean or np.mean(arr)
    std = std or np.std(arr)
    return (arr - mean) / std

@dataclass
class PreprocessedFileInfo:
    dataset_type: str
    data_hash: str
    title: str
    sequence_length: int
    forecast_period: int


class TSDataPreprocessor():
    """
    Time-Series Data Preprocessor (for use with RNN)
    THIS IS ONLY FOR BINARY CLASSIFICATION E.G.
    target = [1, 0, 0, 0]
    so its the first class

    If preprocessed data already exists in the root data folder, it will be loaded, and preprocessing will be skipped
    We will check if it already exists by obtaining a hash of the raw_data dataframe, and comparing it to the hash saved in
    the name of the file. This will mean the same raw data was once passed before.

    @param: custom_norm_functions : a dictionary of column names and their associated normalisation function. Any column
    not specified uses the default normalisation function, which is percentage change, then standardisation.
    
    Shuffling must be done manually or passed as a param to preprocess. Saved datasets are NOT shuffled

    The dataframe caption will be used in the file name when saving the dataset file. Set the dataframe caption with df.style.set_caption("my caption")
    """


    ###### STATIC METHODS ######

    @staticmethod
    def deconstruct_filename(filename: str):
        filename_no_path = os.path.basename(filename)
        filename_no_ext = os.path.splitext(filename_no_path)[0]
        split_filename = filename_no_ext.split("__")
        dataset_type = split_filename[0]
        data_hash = split_filename[1]
        title = split_filename[2]
        sequence_length = int(split_filename[3].split('-')[-1])
        forecast_period = int(split_filename[4].split('-')[-1])
        return PreprocessedFileInfo(dataset_type, data_hash, title, sequence_length, forecast_period)

    @staticmethod
    def get_preprocessed_filename(data_hash: str, df_title: str, sequence_length: int, forecast_period: int):
        return f"{data_hash}__{df_title}__SeqLen-{sequence_length}__Forecast-{forecast_period}.npy"


    @staticmethod
    def load_existing_dataset(raw_data: pd.DataFrame):
        data_hash = TSDataPreprocessor.get_data_hash(raw_data)
        train_folder = os.path.join(os.environ["workspace"], "data", "preprocessed", "train")
        val_folder = os.path.join(os.environ["workspace"], "data", "preprocessed", "validation")
        for file_path in glob.glob(os.path.join(train_folder, "*.npy")):
            file_info = TSDataPreprocessor.deconstruct_filename(file_path)
            if file_info.data_hash == data_hash:
                print("Same hash; this dataset has been preprocessed before. Using old version")
                filename_template = TSDataPreprocessor.get_preprocessed_filename(file_info.data_hash, file_info.title, file_info.sequence_length, file_info.forecast_period)
                train_x = np.load(os.path.join(train_folder, f"train-x__{filename_template}"))
                train_y = np.load(os.path.join(train_folder, f"train-y__{filename_template}"))
                val_x = np.load(os.path.join(val_folder, f"val-x__{filename_template}"))
                val_y = np.load(os.path.join(val_folder, f"val-y__{filename_template}"))
                return Dataset(train_x, train_y, val_x, val_y)

    
    @staticmethod
    def get_col(df: pd.DataFrame, col_name: str) -> Union[None, np.ndarray]:
        col = None
        if col_name is not None:
            col = df[col_name].to_numpy()
        return col

    @staticmethod
    def pct_change(df: pd.DataFrame, custom_pct_change: Dict[str, Callable[[pd.Series], pd.Series]]):
        keys = custom_pct_change.keys()
        for col_name in df:
            new_col = None
            if col_name not in keys:
                new_col = df[col_name].pct_change()
            else:
                new_col = custom_pct_change[col_name](df[col_name])
            df[col_name] = new_col
        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_data_hash(raw_data: pd.DataFrame):
        return hashlib.sha256(pd.util.hash_pandas_object(raw_data, index=True).values).hexdigest()[2:8] # First 6 hex chars


    @staticmethod
    def add_time_data(df: pd.DataFrame, time_col: np.ndarray):
        time_of_day_col = []
        day_of_week_col = []
        week_of_year_col = []
        for val in time_col:
            timestamp = datetime.fromtimestamp(val)
            time_of_day_col.append(timestamp.second + (timestamp.minute * 60) + (timestamp.hour * 60 * 60))
            day_of_week_col.append(timestamp.weekday())
            week_of_year_col.append(timestamp.isocalendar().week)
        
        df["time_of_day"] = minmax_norm(time_of_day_col, minimum=0, maximum=86400)
        df["day_of_week"] = minmax_norm(day_of_week_col, minimum=1, maximum=7)
        df["week_of_year"] = minmax_norm(week_of_year_col, minimum=1, maximum=52)
        return df

    @staticmethod
    def add_target(df: pd.DataFrame, target_col_name: str, forecast_period: int):
        target = []
        raw_target_col = df[target_col_name]
        for index, value in raw_target_col.items():
            try:
                if value < raw_target_col[index + forecast_period]:
                    ## TODO: Enum or globalise this?
                    target.append(0.0) # Buy
                else:
                    # NOTE: This may have a slight bias to selling; if theyre equal target is sell
                    target.append(1.0) # Sell
            except:
                target.append(np.nan)
        df["target"] = target
        df.dropna(inplace=True)
        return df

    @staticmethod
    def make_sequences(df: pd.DataFrame, sequence_length: int):
        sequences: list = [] 
        cur_sequence: Deque = deque(maxlen=sequence_length)
        target_index = df.columns.get_loc("target")
        num_classes = len(df["target"].unique())
        numpy_df = df.to_numpy()
        for index, value in enumerate(numpy_df):
            # Since value is only considered a single value in the sequence (even though itself is an array), to make it a sequence, we encapsulate it in an array so:
            # sequence1 = [[values1], [values2], [values3]]
            val_without_target = np.concatenate((value[:target_index], value[target_index + 1:]))
            cur_sequence.append(val_without_target) # Append all but target to cur_sequence
            if len(cur_sequence) == sequence_length:
                seq = list(cur_sequence)
                target = [0] * num_classes
                target[int(value[target_index])] = 1
                sequences.append([np.array(seq), target]) # value[-1] is the target        
        

        data_x_list = []
        data_y_list = []
        for seq, target in sequences:
            data_x_list.append(seq)
            data_y_list.append(target)
        
        data_x = np.array(data_x_list)
        data_y = np.array(data_y_list)
        return data_x, data_y

    @staticmethod
    def balance_sequences(data_x: np.ndarray, data_y: np.ndarray):
        num_groups = len(data_y[0])
        group_indices: List[List[int]] = [[]] * num_groups
        _, counts = np.unique(data_y, return_counts=True)
        for index, target_tuple in enumerate(data_y):
            target_index = target_tuple.argmax() # e.g. Find the 1 in [0, 0, 1]
            group_indices[target_index].append(index)

        for group, indices in enumerate(group_indices):
            np.random.shuffle(group_indices[group]) # Shuffle removal order
            dif = len(indices) - np.min(counts)
            for i in range(dif):
                index = group_indices[group].pop()
                data_x[index] = np.full(data_x[index].shape, np.nan)
                data_y[index] = np.full(data_y[index].shape, np.nan)
        
        data_x = data_x[~np.isnan(data_x)].reshape(-1, *data_x.shape[1:])
        data_y = data_y[~np.isnan(data_y)].reshape(-1, *data_y.shape[1:])
        return data_x, data_y

    @staticmethod
    def save_dataset(dataset: Dataset, raw_data: pd.DataFrame, df_title: str, sequence_length: int, forecast_period: int):
        folder = os.path.join(os.environ["workspace"], "data", "preprocessed")
        data_hash = TSDataPreprocessor.get_data_hash(raw_data)
        filename = TSDataPreprocessor.get_preprocessed_filename(data_hash, df_title, sequence_length, forecast_period)
        train_folder = os.path.join(folder, "train")
        val_folder = os.path.join(folder, "validation")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        np.save(os.path.join(train_folder, f"train-x__{filename}"), dataset.train_x)
        np.save(os.path.join(train_folder, f"train-y__{filename}"), dataset.train_y)
        np.save(os.path.join(val_folder, f"val-x__{filename}"), dataset.val_x)
        np.save(os.path.join(val_folder, f"val-y__{filename}"), dataset.val_y)


    @staticmethod
    def std_normalisation(df: pd.DataFrame, col_name: str, col_config: ColumnConfig) -> pd.DataFrame:
        df[col_name] = df[col_name].pct_change()

        config_dict = col_config.to_dict()
        mean = config_dict[col_name].get("mean", df[col_name].mean(skipna=True))
        std = config_dict[col_name].get("std", df[col_name].mean(skipna=True))
        if "mean" not in config_dict[col_name] or "std" not in config_dict[col_name]:
            col_config.add_args(col_name, {"mean": mean, "std": std})
        np_data = df[col_name].to_numpy()
        df[col_name] = standardise(np_data, mean, std)
        return df

    @staticmethod
    def ma_std_normalisation(df: pd.DataFrame, col_name: str, period: int, col_config: ColumnConfig) -> pd.DataFrame:
        df[col_name] = df[col_name].rolling(period, center=False).mean()
        df[col_name] = df[col_name].pct_change()

        config_dict = col_config.to_dict()
        mean = config_dict[col_name].get("mean", df[col_name].mean(skipna=True))
        std = config_dict[col_name].get("std", df[col_name].mean(skipna=True))
        if "mean" not in config_dict[col_name] or "std" not in config_dict[col_name]:
            col_config.add_args(col_name, {"mean": mean, "std": std})
        np_data = df[col_name].to_numpy()
        df[col_name] = standardise(np_data, mean, std)
        
        return df

    @staticmethod
    def minmax_normalisation(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        df[col_name] = minmax_norm(df[col_name].to_numpy())
        return df

    @staticmethod
    def time_normalisation(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        time_data = df[col_name].to_numpy()
        df = TSDataPreprocessor.add_time_data(df, time_data)
        df.drop(columns=col_name, inplace=True, errors="ignore")
        return df


    @staticmethod
    def normalisation(df: pd.DataFrame, col_config: ColumnConfig) -> pd.DataFrame:
        config = col_config.to_dict()
        for col_name, value in config.items():
            if col_name == "target": continue
            if value["norm_function"] == NormFunction.STD:
                df = TSDataPreprocessor.std_normalisation(df, col_name, col_config)
            if value["norm_function"] == NormFunction.MA_STD:
                df = TSDataPreprocessor.ma_std_normalisation(df, col_name, value["period"], col_config)
            if value["norm_function"] == NormFunction.MINMAX:
                df = TSDataPreprocessor.minmax_normalisation(df, col_name)
            if value["norm_function"] == NormFunction.TIME:
                df = TSDataPreprocessor.time_normalisation(df, col_name)
        df.dropna(inplace=True)
        return df
        
    ###### OBJECT METHODS ######
        
    def __init__(self, col_config: Optional[ColumnConfig] = None):
        self.col_config = col_config

        self.raw_dynamic_df = pd.DataFrame()
        if col_config is not None:
            columns = col_config.to_dict().keys()
            self.raw_dynamic_df = pd.DataFrame(columns=columns)
        self.dynamic_sequence: Deque = deque()

    def get_current_ATR(self, period: int) -> Union[float, None]:
        """
        Get ATR for current dynamic data (only to be used with dynamic preprocess)
        """
        if len(self.raw_dynamic_df) <= max(2, period):
            return None
        data = self.raw_dynamic_df.iloc[-period:]
        high_low = data['h'] - data['l']
        high_cp = np.abs(data['h'] - data['c'].shift())
        low_cp = np.abs(data['l'] - data['c'].shift())

        df = pd.concat([high_low, high_cp, low_cp], axis=1)
        true_range = np.max(df, axis=1)

        average_true_range = true_range.rolling(period).mean()
        return average_true_range.iloc[-1]
    
    def dynamic_preprocess(self, data_point: dict, seq_len: int) -> Union[np.ndarray, None]:
        if self.col_config is None:
            raise Exception("You must pass col_config to DataUpdater constructor to use the dynamic_preprocess function")
        
        # Add data_point to a raw_dataframe (create one if doesn't exist with correct ordering)
        data_point = {x:[y] for x, y in data_point.items()}
        self.raw_dynamic_df = pd.concat([self.raw_dynamic_df, pd.DataFrame(data_point)])

        # Wait until raw_df is of size (sequence length + max(largest ma_period, 1))  <-- 1 for pct chg (otherwise return None)
        col_config_dict = self.col_config.to_dict()
        ma_periods = [x.get("period", 0) for x in col_config_dict.values()]
        largest_ma_period = max(ma_periods)
        min_required_rows = seq_len + max(largest_ma_period, 1) # Min Required rows to get seq_len rows after normalisation
        if len(self.raw_dynamic_df) <= min_required_rows: # 1 for NaN caused by pct_chg
            return None # We can't make a single sequence yet

        # First run: Take last x (where x is above formula) and normalise the df
        # Turn that into a sequence
        is_first_run = self.dynamic_sequence.maxlen is None and len(self.dynamic_sequence) == 0
        if is_first_run:
            self.dynamic_sequence = deque(maxlen=seq_len)
            last_rows = self.raw_dynamic_df.tail(min_required_rows).copy()
            normalised_df = self.normalisation(last_rows, self.col_config)
            numpy_df = normalised_df.to_numpy()
            for index, value in enumerate(numpy_df):
                self.dynamic_sequence.append(value)

        # Any other run (if sequence exists): take (max(largest ma_period, 1) + 1 (for last point)) of the last data points and normalise
        # Just take the last value and add that to the sequence (removing the first one)
        else:
            min_rows = max(largest_ma_period, 1) + 1
            last_rows = self.raw_dynamic_df.tail(min_rows).copy()
            x = len(last_rows)
            normalised_df = self.normalisation(last_rows, self.col_config)
            last_row = normalised_df.iloc[-1].to_numpy()
            self.dynamic_sequence.append(last_row)
 
        # Return that sequence
        return np.array(self.dynamic_sequence)


    def preprocess(self, raw_data: pd.DataFrame, *,
        col_config: ColumnConfig,
        dataset_metadata: DatasetMetadata
    ) -> Dataset:
        sequence_length = dataset_metadata.sequence_length
        if dataset_metadata.forecast_period.measurement != dataset_metadata.candle_time.measurement:
            raise Exception(f"Difference candle time and forecast period measurements not yet supported. Candle time is in {dataset_metadata.candle_time.measurement}, forecast period is in {dataset_metadata.forecast_period.measurement}. Wanna use it? Implement it BITCH")
        forecast_period = dataset_metadata.forecast_period.multiplier
        train_split = dataset_metadata.train_split
        """
        Notes:
        Somewhere here you will need to save some metadata so you can preprocess new data, not just a dataset.
        E.g. row standard deviations, means, percentiles etc.
        
        Preprocessing Volume:
        Make it a moving average (between 3 and 200 picked by GA)
        Then have it as percent change and standardised
        We best percieve volume as how it changes / slopes. This will best capture this

        All else is standardised
        """
        ## TODO: col_config documentation

        #TODO: THIS DOESN'T CURRENTLY WORK. MEANS AND STD AREN'T ADDED TO COL_CONFIG AMONG OTHER THINGS
        # dataset = self.load_existing_dataset(raw_data)
        # if dataset is not None:
        #     return dataset 

        df = raw_data.copy()
        df_title = df.style.caption or "~"

        # Add Target (target can be added after since its classification)
        df = self.add_target(df, col_config.target_column, forecast_period)

        # Remove time column for later handling
        self.normalisation(df, col_config)


        # Convert into numpy sequences
        # [
        #    [[sequence1], target1]
        #    [[sequence2], target2]
        # ]  
        data_x, data_y = self.make_sequences(df, sequence_length)

        # Balance
        data_x, data_y = self.balance_sequences(data_x, data_y)
        # Split into training and validation

        train_x, val_x = np.split(data_x, [int(train_split * len(data_x))])
        train_y, val_y = np.split(data_y, [int(train_split * len(data_y))])
        dataset = Dataset(train_x, train_y, val_x, val_y)

        # Save data

        self.save_dataset(dataset, raw_data, df_title, sequence_length, forecast_period)

        

        # Shuffle training set 

        # TODO: COME BACK TO SHUFFLE LATER
        # random.shuffle(sequences) # Shuffle sequences to avoid order effects on learning
        print("Values after preprocessing:")
        print(df)
        return Dataset(train_x, train_y, val_x, val_y)