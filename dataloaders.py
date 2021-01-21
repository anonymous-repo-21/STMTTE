import numpy as np

import my_config
import sys
import datasets


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self):
        self.means = {
            "lat": list(map(float, my_config.statistics_config["lat_means"].split(","))),
            "lng": list(map(float, my_config.statistics_config["lng_means"].split(","))),
            "dists": list(map(float, my_config.statistics_config["dist_means"].split(","))),
            "label": list(map(float, my_config.statistics_config["labels_means"].split(",")))
        }
        self.stds = {
            "lat": list(map(float, my_config.statistics_config["lat_stds"].split(","))),
            "lng": list(map(float, my_config.statistics_config["lng_stds"].split(","))),
            "dists": list(map(float, my_config.statistics_config["dist_stds"].split(","))),
            "label": list(map(float, my_config.statistics_config["labels_stds"].split(",")))
        }

    def transform(self, data, dataset_idx, data_type):
        if isinstance(data[0], list):
            return [(np.array(d) - self.means[data_type][dataset_idx]) / self.stds[data_type][dataset_idx] for d in
                    data]
        else:
            return (data - self.means[data_type][dataset_idx]) / self.stds[data_type][dataset_idx]

    # @tf.function(experimental_relax_shapes=True)
    def inverse_transform(self, data, dataset_idx, data_type):
        return (data * self.stds[data_type][dataset_idx]) + self.means[data_type][dataset_idx]


class MyChengduSingleDataLoaderWithEmbedding:
    def __init__(self, args):
        print("Loading data...")
        datasets = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.args = args
        self.scaler = StandardScaler()
        train_files = args.general_config["train_files"].split(",")
        val_files = args.general_config["val_files"].split(",")
        test_files = args.general_config["test_files"].split(",")
        # load regular files
        self.load_regular_files(datasets, test_files, train_files, val_files)
        self.train_lens = [len(datasets["train"][k]["labels"]) for k in datasets["train"].keys()]
        self.val_lens = [len(datasets["val"][k]["labels"]) for k in datasets["val"].keys()]
        self.test_lens = [len(datasets["test"][k]["labels"]) for k in datasets["test"].keys()]
        # make our dataset
        self.train_datasets = [getattr(sys.modules["datasets"], my_config.model_config["dataset"])(
            datasets["train"][idx], int(my_config.general_config["batch_size"])) for idx in range(len(self.train_lens))]
        self.val_datasets = [
            getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][idx],
                                                                                int(
                                                                                    my_config.general_config[
                                                                                        "batch_size"])) for idx in
            range(len(self.val_lens))]
        self.test_datasets = [getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][idx],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
                              for idx in range(len(self.test_lens))]
        # make tf datasets
        # chengdu_test_dataset, chengdu_train_dataset, chengdu_val_dataset, porto_test_dataset, porto_train_dataset, porto_val_dataset = self.make_tf_datasets()
        # Add to list
        # self.train_datasets = []
        # self.val_datasets = []
        # self.test_datasets = []
        # self.train_datasets.append(chengdu_train_dataset)
        # self.train_datasets.append(porto_train_dataset)
        # self.val_datasets.append(chengdu_val_dataset)
        # self.val_datasets.append(porto_val_dataset)
        # self.test_datasets.append(chengdu_test_dataset)
        # self.test_datasets.append(porto_test_dataset)
        # shuffle and make batches
        self.train_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.train_datasets)
            )
        )
        self.val_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.val_datasets)
            )
        )
        self.test_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.test_datasets)
            )
        )
        print("Loading data finished.")

    def load_regular_files(self, datasets, test_files, train_files, val_files):
        # filter those uncommon range data
        # filter_range = [
        #     (14, 141), (18, 81)
        # ]
        filter_range = (14, 141)
        for idx in range(len(train_files)):
            datasets["train"][idx] = {}
            train_lats = np.load(f"{train_files[idx]}-lats.npy", allow_pickle=True)
            train_lngs = np.load(f"{train_files[idx]}-lngs.npy", allow_pickle=True)
            train_timeIDs = np.load(f"{train_files[idx]}-timeID.npy", allow_pickle=True)
            train_weekIDs = np.load(f"{train_files[idx]}-weekID.npy", allow_pickle=True)
            train_labels = np.load(f"{train_files[idx]}-labels.npy", allow_pickle=True)
            train_dists = np.load(f"{train_files[idx]}-dist.npy", allow_pickle=True)
            train_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in train_lats]
            train_lats = train_lats[train_length_mask]
            train_lngs = train_lngs[train_length_mask]
            train_timeIDs = train_timeIDs[train_length_mask]
            train_weekIDs = train_weekIDs[train_length_mask]
            train_dists = train_dists[train_length_mask]
            train_labels = train_labels[train_length_mask]
            datasets["train"][idx]["lats"] = self.scaler.transform(
                train_lats
                , 0,
                "lat")
            datasets["train"][idx]["lngs"] = self.scaler.transform(
                train_lngs
                , 0,
                "lng")
            datasets["train"][idx]["timeIDs"] = train_timeIDs
            datasets["train"][idx]["weekIDs"] = train_weekIDs
            datasets["train"][idx]["dists"] = train_dists
            datasets["train"][idx]["labels"] = self.scaler.transform(
                train_labels
                , 0, "label")
            datasets["train"][idx]["dists"] = self.scaler.transform(
                train_dists
                , 0, "dists")

            if idx == 0:
                datasets["val"][idx] = {}
                val_lats = np.load(f"{val_files[idx]}-lats.npy", allow_pickle=True)
                val_lngs = np.load(f"{val_files[idx]}-lngs.npy", allow_pickle=True)
                val_labels = np.load(f"{val_files[idx]}-labels.npy", allow_pickle=True)
                val_timeIDs = np.load(f"{val_files[idx]}-timeID.npy", allow_pickle=True)
                val_weekIDs = np.load(f"{val_files[idx]}-weekID.npy", allow_pickle=True)
                val_dists = np.load(f"{val_files[idx]}-dist.npy", allow_pickle=True)
                val_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in val_lats]
                val_lats = val_lats[val_length_mask]
                val_lngs = val_lngs[val_length_mask]
                val_timeIDs = val_timeIDs[val_length_mask]
                val_weekIDs = val_weekIDs[val_length_mask]
                val_dists = val_dists[val_length_mask]
                val_labels = val_labels[val_length_mask]
                datasets["val"][idx]["lats"] = self.scaler.transform(
                    val_lats, idx, "lat")
                datasets["val"][idx]["lngs"] = self.scaler.transform(
                    val_lngs, idx, "lng")
                datasets["val"][idx]["labels"] = self.scaler.transform(
                    val_labels, idx,
                    "label")
                datasets["val"][idx]["timeIDs"] = val_timeIDs
                datasets["val"][idx]["weekIDs"] = val_weekIDs
                datasets["val"][idx]["dists"] = self.scaler.transform(
                    val_dists, 0,
                    "dists")
                datasets["test"][idx] = {}
                test_lats = np.load(f"{test_files[idx]}-lats.npy", allow_pickle=True)
                test_lngs = np.load(f"{test_files[idx]}-lngs.npy", allow_pickle=True)
                test_labels = np.load(f"{test_files[idx]}-labels.npy", allow_pickle=True)
                test_timeIDs = np.load(f"{test_files[idx]}-timeID.npy", allow_pickle=True)
                test_weekIDs = np.load(f"{test_files[idx]}-weekID.npy", allow_pickle=True)
                test_dists = np.load(f"{test_files[idx]}-dist.npy", allow_pickle=True)
                test_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in test_lats]
                test_lats = test_lats[test_length_mask]
                test_lngs = test_lngs[test_length_mask]
                test_timeIDs = test_timeIDs[test_length_mask]
                test_weekIDs = test_weekIDs[test_length_mask]
                test_dists = test_dists[test_length_mask]
                test_labels = test_labels[test_length_mask]
                datasets["test"][idx]["lats"] = self.scaler.transform(
                    test_lats, idx,
                    "lat")
                datasets["test"][idx]["lngs"] = self.scaler.transform(
                    test_lngs, idx,
                    "lng")
                datasets["test"][idx]["timeIDs"] = test_timeIDs
                datasets["test"][idx]["weekIDs"] = test_weekIDs
                datasets["test"][idx]["dists"] = self.scaler.transform(
                    test_dists, 0,
                    "dists")
                datasets["test"][idx]["labels"] = self.scaler.transform(
                    test_labels, idx,
                    "label")


class MyPortoSingleDataLoaderWithEmbedding:
    def __init__(self, args):
        print("Loading data...")
        datasets = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.args = args
        self.scaler = StandardScaler()
        train_files = args.general_config["train_files"].split(",")
        val_files = args.general_config["val_files"].split(",")
        test_files = args.general_config["test_files"].split(",")
        # load regular files
        self.load_regular_files(datasets, test_files, train_files, val_files)
        self.train_lens = [len(datasets["train"][k]["labels"]) for k in datasets["train"].keys()]
        self.val_lens = [len(datasets["val"][k]["labels"]) for k in datasets["val"].keys()]
        self.test_lens = [len(datasets["test"][k]["labels"]) for k in datasets["test"].keys()]
        # make our dataset
        self.train_datasets = [getattr(sys.modules["datasets"], my_config.model_config["dataset"])(
            datasets["train"][idx], int(my_config.general_config["batch_size"])) for idx in range(len(self.train_lens))]
        self.val_datasets = [
            getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][idx],
                                                                                int(
                                                                                    my_config.general_config[
                                                                                        "batch_size"])) for idx in
            range(len(self.val_lens))]
        self.test_datasets = [getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][idx],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
                              for idx in range(len(self.test_lens))]
        # make tf datasets
        # chengdu_test_dataset, chengdu_train_dataset, chengdu_val_dataset, porto_test_dataset, porto_train_dataset, porto_val_dataset = self.make_tf_datasets()
        # Add to list
        # self.train_datasets = []
        # self.val_datasets = []
        # self.test_datasets = []
        # self.train_datasets.append(chengdu_train_dataset)
        # self.train_datasets.append(porto_train_dataset)
        # self.val_datasets.append(chengdu_val_dataset)
        # self.val_datasets.append(porto_val_dataset)
        # self.test_datasets.append(chengdu_test_dataset)
        # self.test_datasets.append(porto_test_dataset)
        # shuffle and make batches
        self.train_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.train_datasets)
            )
        )
        self.val_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.val_datasets)
            )
        )
        self.test_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.test_datasets)
            )
        )
        print("Loading data finished.")

    def load_regular_files(self, datasets, test_files, train_files, val_files):
        # filter those uncommon range data
        # filter_range = [
        #     (14, 141), (18, 81)
        # ]
        filter_range = (14, 141)
        for idx in range(len(train_files)):
            datasets["train"][idx] = {}
            train_lats = np.load(f"{train_files[idx]}-lats.npy", allow_pickle=True)
            train_lngs = np.load(f"{train_files[idx]}-lngs.npy", allow_pickle=True)
            train_timeIDs = np.load(f"{train_files[idx]}-timeID.npy", allow_pickle=True)
            train_weekIDs = np.load(f"{train_files[idx]}-weekID.npy", allow_pickle=True)
            train_labels = np.load(f"{train_files[idx]}-labels.npy", allow_pickle=True)
            train_dists = np.load(f"{train_files[idx]}-dist.npy", allow_pickle=True)
            train_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in train_lats]
            train_lats = train_lats[train_length_mask]
            train_lngs = train_lngs[train_length_mask]
            train_timeIDs = train_timeIDs[train_length_mask]
            train_weekIDs = train_weekIDs[train_length_mask]
            train_dists = train_dists[train_length_mask]
            train_labels = train_labels[train_length_mask]
            datasets["train"][idx]["lats"] = self.scaler.transform(
                train_lats
                , 1,
                "lat")
            datasets["train"][idx]["lngs"] = self.scaler.transform(
                train_lngs
                , 1,
                "lng")
            datasets["train"][idx]["timeIDs"] = train_timeIDs
            datasets["train"][idx]["weekIDs"] = train_weekIDs
            datasets["train"][idx]["dists"] = self.scaler.transform(
                train_dists
                , 1, "dists")

            datasets["train"][idx]["labels"] = self.scaler.transform(
                train_labels
                , 1, "label")

            if idx == 0:
                datasets["val"][idx] = {}
                val_lats = np.load(f"{val_files[idx]}-lats.npy", allow_pickle=True)
                val_lngs = np.load(f"{val_files[idx]}-lngs.npy", allow_pickle=True)
                val_labels = np.load(f"{val_files[idx]}-labels.npy", allow_pickle=True)
                val_timeIDs = np.load(f"{val_files[idx]}-timeID.npy", allow_pickle=True)
                val_weekIDs = np.load(f"{val_files[idx]}-weekID.npy", allow_pickle=True)
                val_dists = np.load(f"{val_files[idx]}-dist.npy", allow_pickle=True)
                val_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in val_lats]
                val_lats = val_lats[val_length_mask]
                val_lngs = val_lngs[val_length_mask]
                val_timeIDs = val_timeIDs[val_length_mask]
                val_weekIDs = val_weekIDs[val_length_mask]
                val_dists = val_dists[val_length_mask]
                val_labels = val_labels[val_length_mask]
                datasets["val"][idx]["lats"] = self.scaler.transform(
                    val_lats, 1, "lat")
                datasets["val"][idx]["lngs"] = self.scaler.transform(
                    val_lngs, 1, "lng")
                datasets["val"][idx]["labels"] = self.scaler.transform(
                    val_labels, 1,
                    "label")
                datasets["val"][idx]["timeIDs"] = val_timeIDs
                datasets["val"][idx]["weekIDs"] = val_weekIDs
                datasets["val"][idx]["dists"] = self.scaler.transform(
                    val_dists, 1,
                    "dists")
                datasets["test"][idx] = {}
                test_lats = np.load(f"{test_files[idx]}-lats.npy", allow_pickle=True)
                test_lngs = np.load(f"{test_files[idx]}-lngs.npy", allow_pickle=True)
                test_labels = np.load(f"{test_files[idx]}-labels.npy", allow_pickle=True)
                test_timeIDs = np.load(f"{test_files[idx]}-timeID.npy", allow_pickle=True)
                test_weekIDs = np.load(f"{test_files[idx]}-weekID.npy", allow_pickle=True)
                test_dists = np.load(f"{test_files[idx]}-dist.npy", allow_pickle=True)
                test_length_mask = [filter_range[1] >= len(x) >= filter_range[0] for x in test_lats]
                test_lats = test_lats[test_length_mask]
                test_lngs = test_lngs[test_length_mask]
                test_timeIDs = test_timeIDs[test_length_mask]
                test_weekIDs = test_weekIDs[test_length_mask]
                test_dists = test_dists[test_length_mask]
                test_labels = test_labels[test_length_mask]
                datasets["test"][idx]["lats"] = self.scaler.transform(
                    test_lats, 1,
                    "lat")
                datasets["test"][idx]["lngs"] = self.scaler.transform(
                    test_lngs, 1,
                    "lng")
                datasets["test"][idx]["timeIDs"] = test_timeIDs
                datasets["test"][idx]["weekIDs"] = test_weekIDs
                datasets["test"][idx]["dists"] = self.scaler.transform(
                    test_dists, 1,
                    "dists")
                datasets["test"][idx]["labels"] = self.scaler.transform(
                    test_labels, 1,
                    "label")
