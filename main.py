from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import Dataset
from model import *

para_dict = {"l1": [True, False],
             "l2": [True, False],
             "hidden_dim": [256, 128, 64],
             "deep_dim": [32, 16, 8],
             "coefficient": [0.01, 0.1, 0.5],
             "drop_rate": [0.1, 0.2, 0.5]}


def run_cv(feature_file, survival_file, cv: int = 5, ):
    """
    train and test the performance with cross validation
    :param cv: number of cross validation
    :param feature_file: csv file path
    :param survival_file: csv file path
    :return:  
    """
    # get the training and test dataset
    my_dataset = Dataset(feature_file, survival_file)
    my_dataset.split_dataset(cv)
    zipped_dataset = my_dataset.to_array()
    for i, v in enumerate(zipped_dataset):
        train_array = v[0]
        test_array = v[1]
        x_input = train_array[0]
        y_input = train_array[1]
        x_test = test_array[0]
        y_test = test_array[1]

        feature_dim = x_input.shape[1]
        sample_num = x_input.shape[0]
        my_model = Cox_autoencoder(feature_dim,
                                   l1=True,
                                   l2=False,
                                   coefficient=0.01,
                                   drop_rate=0.1,
                                   hidden_dim=256,
                                   deep_dim=16,
                                   activation=tf.nn.relu)
        # train the model
        trained_model = train_model(my_model,
                                    x_input,
                                    y_input,
                                    batch_size=sample_num,
                                    global_step=0,
                                    optimizer=tf.optimizers.Adam(learning_rate=0.001),
                                    num_epochs=1000)
        print(trained_model)
        print("training is over")

        # test the performance
        output, deep_feature, hazard, event_batch, r_batch = trained_model(x_test,
                                                                           y_test)
        print(output)
        print(deep_feature)
        print(hazard)
        print(event_batch)
        print(r_batch)
    return 0


def run_train_test_splitting(feature_file, survival_file, test_frac: float = 0.33):
    """
    workflow: training on training dataset and test the performance on test dataset.
    :param test_frac: the fraction of test dataset
    :param feature_file: file_path of sample features,csv format
    :param survival_file: file path of sample survival data, csv format
    :return: 0
    """
    # get the training and test dataset
    my_dataset = Dataset(feature_file, survival_file)
    my_dataset.train_test_split(test_frac)  # test:1/3

    index_lst = my_dataset.get_sample_name()
    index_tup = index_lst[0]
    train_sample_name = index_tup[0]
    test_sample_name = index_tup[1]

    dataset_lst = my_dataset.to_array()
    dataset_tup = dataset_lst[0]
    x_train = dataset_tup[0]
    y_train = dataset_tup[1]
    x_test = dataset_tup[2]
    y_test = dataset_tup[3]

    feature_dim = x_train.shape[1]
    sample_num = x_train.shape[0]

    my_model = Cox_autoencoder(feature_dim,
                               l1=True,
                               l2=False,
                               coefficient=0.01,
                               drop_rate=0.1,
                               hidden_dim=256,
                               deep_dim=16,
                               activation=tf.nn.relu)

    # traditional cox model
    # my_model = Cox()
    # lasso cox model
    # my_model = Cox(l1=True)

    # ridge cox model l1=True,
    # my_model = Cox(l2=True)

    # my_model = Cox_nnet(l1=True,
    #                     l2=False,
    #                     coefficient=0.01,
    #                     drop_rate=0.1,
    #                     hidden_dim=256,
    #                     deep_dim=16,
    #                     activation=tf.nn.relu)
    # train the model
    trained_model = train_model(my_model,
                                x_train,
                                y_train,
                                batch_size=sample_num,
                                global_step=0,
                                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                                num_epochs=2000)
    print("training is over.")

    # for ae model
    output, deep_feature, hazard, event_batch, r_batch = trained_model(x_train,
                                                                       y_train)

    # for cox model
    # hazard, event_batch, r_batch = trained_model(x_train, y_train)
    # train_df = get_hazard_df(hazard, train_sample_name)

    # for nnet model
    # deep_feature, hazard, event_batch, r_batch = trained_model(x_train,
    #                                                            y_train)

    # for ae and nnet model
    train_df = get_deep_feature_hazard_df(deep_feature,
                                          hazard,
                                          train_sample_name)
    train_df.to_csv("./data/dss_results/ae_train_deep_dss.csv", index=False)
    # deep features of test data

    # for cox model
    # hazard, event_batch, r_batch = trained_model(x_test, y_test)

    # for ae model
    output, deep_feature, hazard, event_batch, r_batch = trained_model(x_test,
                                                                       y_test)

    # for nnet model
    # deep_feature, hazard, event_batch, r_batch = trained_model(x_test,
    #                                                            y_test)
    # for cox model
    # test_df = get_hazard_df(hazard, test_sample_name)

    # for ae and nnet models
    test_df = get_deep_feature_hazard_df(deep_feature,
                                         hazard,
                                         test_sample_name)
    test_df.to_csv("./data/dss_results/ae_test_deep_dss.csv", index=False)
    return 0


def run_whole(feature_file, survival_file):
    # read the data
    print("read the data")
    my_dataset = Dataset(feature_file, survival_file)
    x_input, y_input = my_dataset.to_array()
    # model
    print("construct the model")
    feature_dim = x_input.shape[1]
    sample_num = x_input.shape[0]
    my_model = Cox_autoencoder(feature_dim,
                               l1=True,
                               l2=False,
                               coefficient=0.01,
                               drop_rate=0.1,
                               hidden_dim=256,
                               deep_dim=16,
                               activation=tf.nn.relu)
    # train the model
    print("train the model")
    trained_model = train_model(my_model,
                                x_input,
                                y_input,
                                batch_size=sample_num,
                                global_step=0,
                                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                                num_epochs=2000)

    print(trained_model)
    print("training is over")
    trained_model.save_weights("./data/os_results/cox_ae_weights_train_GSE39582.ckpt")
    print("training is over and the weights are saved.")
    output, deep_feature, hazard, event_batch, r_batch = trained_model(x_input,
                                                                       y_input)
    whole_index = my_dataset.get_sample_name()
    whole_deep_df = get_deep_feature_hazard_df(deep_feature,
                                               hazard,
                                               whole_index)
    whole_deep_df.to_csv("./data/os_results/ae_whole_deep_GSE39582.csv", index=False)
    return 0


def plot_tSNE(m_file):
    """
    :type m_file: csv file with sample names as rows and features as columns
    """
    # read the file into dataframe
    m_df = pd.read_csv(m_file, index_col=0)
    # print(m_df.shape)
    # print(m_df.head())
    deep_features = np.array(m_df.iloc[:, :-1])  # last column:prognosis_index
    print("deep_features.shape")
    print(deep_features.shape)
    hazard = list(m_df["prognosis_index"])
    z = deep_features
    z_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(z)
    color = hazard  # color based on the value of hazard
    plt.figure(figsize=(12, 10))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=80, c=color, cmap="RdBu_r")
    plt.tick_params(labelsize=20)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    features_file = "./data/cox_selected_features.csv"
    # survival_file_os = "./data/os.csv"
    survival_file_dss = "./data/dss.csv"

    # GSE39582
    # features_file = "./data/cox_selected_features_GSE39582.csv"
    # survival_file_os = "./data/GSE39582_os.csv"
    #run_whole(features_file, survival_file_os)
    # run_train_test_splitting(features_file, survival_file_dss)
    # plot_tSNE("./data/dss_results/ae_test_deep_dss.csv")

    # survival_file_pfi = "./data/pfi.csv"
    #run_whole(features_file, survival_file_os)
    #run_train_test_splitting(features_file, survival_file_os)
    #run_train_test_splitting(mRNA_features_file, survival_file_os)
    #plot_tSNE("./data/os_results/ae_test_deep.csv")

    m_data = Dataset(features_file, survival_file_dss)
    print(m_data.get_sample_num())
    print(m_data.get_event_number())

