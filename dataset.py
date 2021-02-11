import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class Dataset(object):
    """
    Handle the dataset: Read the data from files,
    merge and split the dataset for downstream analysis.
    """

    def __init__(self, m_feature_file, m_survival_file):
        """ Read the datasets from files.
        Args：
            feature_file: csv file,first line: sample, feature1,feature2,...
            survival_file: csv file,first line: sample, event,event_time
        """
        self.__feature_df = pd.read_csv(m_feature_file)
        self.__survival_df = pd.read_csv(m_survival_file)
        self.__df = self.__merge()
        self.__transform_data()
        self.__dataset = []  # list of train/test tuple
        self.__is_splitted = False

    def get_event_number(self):
        return len(self.__df[self.__df.iloc[:, -2] == 1])

    def get_event_time(self):
        return self.__df.iloc[:, -1]

    def get_feature_num(self):
        return self.__df.shape[1]

    def get_sample_num(self):
        return len(self.__df)

    def get_feature_statistics(self):
        return self.__df.iloc[:, :-2].describe()

    def get_dim(self):
        return self.__df.shape

    def __merge(self):
        """merge the two datasets into one with the sample id.
        """
        if list(self.__feature_df)[0] == list(self.__survival_df)[0]:
            merge_item = list(self.__feature_df)[0]
            merged_df = pd.merge(self.__feature_df, self.__survival_df, on=merge_item)
            merged_df = merged_df.dropna(axis=0, how='any') # important
            return merged_df
        else:
            print("the first feature of the two dataframes should be the same.")
            return pd.DataFrame()

    def __transform_data(self):
        """
        Transform the dataset: index: sample id ;
                               column: feature1,feature2,...,event,event_time
        """
        new_index = pd.Series([x + "A" for x in self.__df.iloc[:, 0]])
        self.__df.set_index(new_index, inplace=True)
        self.__df = self.__df.iloc[:, 1:]  # remove the first column
        return 0

    def _split(self, m_df, cv: int = 5):
        """function of stratified sampling"""
        num = len(m_df)
        fold_num = int(num / cv)
        _train_list = []
        _test_list = []
        for i in range(cv):
            start = i * fold_num
            end = (i + 1) * fold_num
            if i == (cv - 1):  # last one
                end = num

            _test_df = m_df[start:end]
            _test_list.append(_test_df)
            _train_df = m_df.drop(index=_test_df.index)  # remove the test rows
            _train_list.append(_train_df)

        return _train_list, _test_list

    def cv_split(self, cv: int = 5, random_state: int = 1):
        """ stratified sampling the dataset into train and test for cross validation.
            @ param dataset: the whole dataset for splitting
            @ param cv: times of cross validation
            @ param random_state: seed for sklearn.utils shuffle function
        """
        if self.__is_splitted:
            print("already split the dataset.")
            return -1
        event = list(self.__df)[-2]  # event name
        event_df = self.__df[self.__df[event] == 1]
        censored_df = self.__df[self.__df[event] == 0]

        event_df = shuffle(event_df, random_state=random_state)
        censored_df = shuffle(censored_df, random_state=random_state)

        # event and censored data separation
        train_event_list, test_event_list = self._split(event_df, cv)
        train_censored_list, test_censored_list = self._split(censored_df, cv)

        for i in range(len(train_event_list)):
            train_df = pd.concat([train_event_list[i], train_censored_list[i]])
            test_df = pd.concat([test_event_list[i], test_censored_list[i]])

            train_df = shuffle(train_df, random_state=random_state)
            test_df = shuffle(test_df, random_state=random_state)
            self.__dataset.append((train_df, test_df))
        self.__is_splitted = True
        return 0

    def train_test_split(self, test_frac: float = 0.33, random_state: int = 1):
        if self.__is_splitted:
            print("already split the dataset.")
            return -1
        event = list(self.__df)[-2]  # event name
        event_df = self.__df[self.__df[event] == 1]
        censored_df = self.__df[self.__df[event] == 0]
        # sampling
        test_event_df = event_df.sample(frac=test_frac)
        print(test_event_df.head())
        train_event_df = event_df.drop(index=test_event_df.index)
        test_censored_df = censored_df.sample(frac=test_frac)
        train_censored_df = censored_df.drop(index=test_censored_df.index)

        train_df = train_censored_df.append(train_event_df)
        print(train_df.head())
        test_df = test_censored_df.append(test_event_df)

        train_df = shuffle(train_df, random_state=random_state)
        test_df = shuffle(test_df, random_state=random_state)
        self.__dataset.append((train_df, test_df))
        self.__is_splitted = True
        return 0

    def to_array(self):
        """
        transform the dataframe into numpy array
        """
        if self.__is_splitted:
            m_dataset = []
            for v in self.__dataset:
                train_df = v[0]
                test_df = v[1]
                # last two columns: event  event_time ( eg PFI  PFI.time)
                x_train_df = train_df.iloc[:, :-2]
                y_train_df = train_df.iloc[:, -2:]
                x_test_df = test_df.iloc[:, :-2]
                y_test_df = test_df.iloc[:, -2:]
                # convert to numpy array
                x_train = np.array(x_train_df)
                y_train = np.array(y_train_df)
                x_test = np.array(x_test_df)
                y_test = np.array(y_test_df)

                m_dataset.append((x_train, y_train, x_test, y_test))
            return m_dataset
        else:
            x_df = self.__df.iloc[:, :-2]
            y_df = self.__df.iloc[:, -2:]

            x_array = np.array(x_df, dtype=np.float)
            y_array = np.array(y_df, dtype=np.float)
            return x_array, y_array

    def get_sample_name(self):
        """get the sample id of the self.dataset
           return the index of all sample or index list for splitted dataset
        """
        if self.__is_splitted:
            index_lst = []
            for v in self.__dataset:
                train_df = v[0]
                test_df = v[1]
                index_lst.append((train_df.index, test_df.index))
            return index_lst
        else:
            return self.__df.index


if __name__ == "__main__":
    feature_file = "./data/cox_selected_features.csv"
    survival_file_os = "./data/os.csv"
    my_dataset = Dataset(feature_file, survival_file_os)
    print(my_dataset.get_event_number())
