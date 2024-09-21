from pywebio import *
from pywebio.input import *
from pywebio.output import *
import secretflow as sf
import pandas as pd
import tempfile
from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.data.split import train_test_split
from secretflow.security.aggregation import SecureAggregator
from secretflow.ml.nn import FLModel
from matplotlib import pyplot as plt
from pywebio.input import file_upload
from io import BytesIO
import jax
import secretflow.distributed as sfd
class myFLModel:
    def __init__(self) -> None:
        sf.shutdown()
        print('The version of SecretFlow: {}'.format(sf.__version__))

        sf.init(['alice','bob','charlie'], address='local')
        alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
        self.fed_model = None
        self.history = None
        self.alice = alice
        self.bob = bob
        self.charlie = charlie
        self.spu = spu
        self.prediction = None

    def create_regression_model(self,input_shape="(1,)", name='model'):
        def create_model():
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),  # 输入年龄特征
                    layers.Dense(64, activation="relu"),  # 隐藏层
                    layers.Dense(64, activation="relu"),  # 可以添加更多隐藏层
                    layers.Dense(64, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1)  # 输出层，回归任务没有激活函数，输出一个连续值
                ]
            )
            model.compile(
                loss='mean_squared_error',  # 回归任务的损失函数
                optimizer='adam',  # 优化器
                metrics=['mae','mse']  # 额外的评价指标，例如平均绝对误差（MAE）
            )
            return model

        return create_model


    def create_and_train_model(self):

        alldata_df_1 = pd.read_csv("data/1.csv")
        alldata_df_2 = pd.read_csv("data/2.csv")

        h_alice_df = alldata_df_1.loc[:]
        h_bob_df = alldata_df_2.loc[:]

        _, h_alice_path = tempfile.mkstemp()
        _, h_bob_path = tempfile.mkstemp()
        h_alice_df.to_csv(h_alice_path, index=False)
        h_bob_df.to_csv(h_bob_path, index=False)

        path_dict = {self.alice: h_alice_path, self.bob: h_bob_path}

        aggregator = PlainAggregator(self.charlie)
        comparator = PlainComparator(self.charlie)

        hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)

        label_x = hdf["x"]
        data_x = hdf.drop(columns="y")

        train_data_x, test_data_x = train_test_split(
            data_x, train_size=0.8, shuffle=True, random_state=1024
        )

        label_y = hdf["y"]
        data_y = hdf.drop(columns="x")

        train_data_y, test_data_y = train_test_split(
            data_y, train_size=0.8, shuffle=True, random_state=512
        )

        input_shape = (1,)
        model = self.create_regression_model(input_shape)

        device_list = [self.alice, self.bob]

        secure_aggregator = SecureAggregator(self.charlie, [self.alice, self.bob])


        self.fed_model = FLModel(
            server=self.charlie,
            device_list=device_list,
            model=model,
            aggregator=secure_aggregator,
            strategy="fed_prox",
            backend="tensorflow",
        )

        self.history = self.fed_model.fit(
            train_data_x,
            train_data_y,
            validation_data=(test_data_x, test_data_y),
            epochs=10,
            sampler_method="batch",
            batch_size=10,
            aggregate_freq=1,
        ) 

    def show(self):
        # # Draw mse values for training & validation
        plt.plot(self.history["global_history"]['mse'])
        plt.plot(self.history["global_history"]['val_mse'])
        plt.title('FLModel mse')
        plt.ylabel('mse')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.savefig("figure/mse.png")


        # # Draw mae for training & validation
        plt.plot(self.history["global_history"]['mae'])
        plt.plot(self.history["global_history"]['val_mae'])
        plt.title('FLModel mae')
        plt.ylabel('Mae')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.savefig("figure/mae.png")


        put_image(open('figure/mae.png', 'rb').read())
        put_image(open('figure/mse.png', 'rb').read())
        

    def predict(self,csv):
        with BytesIO(csv['content']) as f:
            df = pd.read_csv(f)
        size = len(df)
        h_alice_df = df.loc[:size//2]
        h_bob_df = df.loc[size//2:2*size//2]

        _, h_alice_path = tempfile.mkstemp()
        _, h_bob_path = tempfile.mkstemp()
        h_alice_df.to_csv(h_alice_path, index=False)
        h_bob_df.to_csv(h_bob_path, index=False)

        path_dict = {self.alice: h_alice_path, self.bob: h_bob_path}

        aggregator = PlainAggregator(self.charlie)
        comparator = PlainComparator(self.charlie)

        hdf_data = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)

        self.prediction = self.fed_model.predict(  
            x=hdf_data,  
            batch_size=10, 
            label_decoder=None, 
            sampler_method='batch',  
            random_seed=1234,  
            dataset_builder=None  
        ) 
        flatten_val, tree = jax.tree_util.tree_flatten(self.prediction)
        cholesterol_list = []
        for i in flatten_val:
            j = sfd.get(i.data)
            numpy_arrays = [tensor.numpy()[0] for tensor in j] 
            for cholesterol in numpy_arrays:
                cholesterol_list.append(cholesterol)

        result = []
        for i in range(len(df)):
            res = []
            res.append(df.loc[i,'age'])
            res.append(cholesterol_list[i]) 
            result.append(res)
        return result   
            
def on_click_start(model):
    popup('training')
    model.create_and_train_model()
    popup('done')

def on_click_show(model):
    model.show()

def on_click_predict(model):
    file = file_upload(label="Select a csv file:")
    res = model.predict(file)
    res.insert(0,["age","cholesterol"])
    put_table(
        res
    )


def myfirstpage():
    model = myFLModel()
    put_column([None,
    put_row([None,
        put_buttons(["----------Train----------"], onclick=[lambda: on_click_start(model)]),
        None
    ],size='40% 300px 60%'),
    None,
    put_row([None,
        put_buttons(["----------Result----------"], onclick=[lambda: on_click_show(model)]),
        None
    ],size='40% 300px 60%'),
    None,
    put_row([None,
        put_buttons(["----------Predict----------"], onclick=[lambda: on_click_predict(model)]),
        None
    ],size='40% 300px 60%'),
    None
    ],size='80px 50px 100px 50px 100px 50px 200px')        
        
if __name__ == '__main__':
    start_server(myfirstpage,port=8083,auto_open_webbrowser=True)
