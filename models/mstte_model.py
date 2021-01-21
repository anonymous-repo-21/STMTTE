import tensorflow as tf

import my_config

class MSMTTEGRUAtt64Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 64)
        self.week_embedding = tf.keras.layers.Embedding(7, 64)
        self.spatial_lstm = tf.keras.layers.GRU(64)
        self.dist_lstm = tf.keras.layers.GRU(64)
        self.hour_temporal_lstm = tf.keras.layers.GRU(64)
        self.week_temporal_lstm = tf.keras.layers.GRU(64)
        self.attention_weights = tf.keras.layers.Dense(4, activation="relu")
        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(512, activation="relu")
        self.fc3 = tf.keras.layers.Dense(256, activation="relu")
        self.fc4 = tf.keras.layers.Dense(64, activation="relu")
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            dist_data = tf.expand_dims(each_input[:, 4:], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            dist_features = self.dist_lstm(dist_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            spatial_features = tf.expand_dims(spatial_features, axis=2)
            dist_features = tf.expand_dims(dist_features, axis=2)
            hour_temporal_features = tf.expand_dims(hour_temporal_features, axis=2)
            week_temporal_features = tf.expand_dims(week_temporal_features, axis=2)
            all_features = tf.concat([spatial_features, hour_temporal_features, week_temporal_features, dist_features], axis=2)
            scores = tf.nn.softmax(self.attention_weights(all_features), axis=2)
            scored_features = tf.reduce_sum(all_features * scores, axis=2)
            x_shortcut = scored_features
            x = self.fc1(scored_features)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            # x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x + x_shortcut)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output

class MSMTTELSTMAtt64Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 64)
        self.week_embedding = tf.keras.layers.Embedding(7, 64)
        self.spatial_lstm = tf.keras.layers.LSTM(64)
        self.dist_lstm = tf.keras.layers.LSTM(64)
        self.hour_temporal_lstm = tf.keras.layers.LSTM(64)
        self.week_temporal_lstm = tf.keras.layers.LSTM(64)
        self.attention_weights = tf.keras.layers.Dense(4, activation="relu")
        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(512, activation="relu")
        self.fc3 = tf.keras.layers.Dense(256, activation="relu")
        self.fc4 = tf.keras.layers.Dense(64, activation="relu")
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            dist_data = tf.expand_dims(each_input[:, 4:], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            dist_features = self.dist_lstm(dist_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            spatial_features = tf.expand_dims(spatial_features, axis=2)
            dist_features = tf.expand_dims(dist_features, axis=2)
            hour_temporal_features = tf.expand_dims(hour_temporal_features, axis=2)
            week_temporal_features = tf.expand_dims(week_temporal_features, axis=2)
            all_features = tf.concat([spatial_features, hour_temporal_features, week_temporal_features, dist_features], axis=2)
            scores = tf.nn.softmax(self.attention_weights(all_features), axis=2)
            scored_features = tf.reduce_sum(all_features * scores, axis=2)
            x_shortcut = scored_features
            x = self.fc1(scored_features)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            # x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x + x_shortcut)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output

class MSMTTEBiLSTMAtt64Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 64)
        self.week_embedding = tf.keras.layers.Embedding(7, 64)
        self.spatial_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode="sum")
        self.dist_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode="sum")
        self.hour_temporal_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode="sum")
        self.week_temporal_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode="sum")
        self.attention_weights = tf.keras.layers.Dense(4, activation="relu")
        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(512, activation="relu")
        self.fc3 = tf.keras.layers.Dense(256, activation="relu")
        self.fc4 = tf.keras.layers.Dense(64, activation="relu")
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            dist_data = tf.expand_dims(each_input[:, 4:], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            dist_features = self.dist_lstm(dist_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            spatial_features = tf.expand_dims(spatial_features, axis=2)
            dist_features = tf.expand_dims(dist_features, axis=2)
            hour_temporal_features = tf.expand_dims(hour_temporal_features, axis=2)
            week_temporal_features = tf.expand_dims(week_temporal_features, axis=2)
            all_features = tf.concat([spatial_features, hour_temporal_features, week_temporal_features, dist_features], axis=2)
            scores = tf.nn.softmax(self.attention_weights(all_features), axis=2)
            scored_features = tf.reduce_sum(all_features * scores, axis=2)
            x_shortcut = scored_features
            x = self.fc1(scored_features)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            # x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x + x_shortcut)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output