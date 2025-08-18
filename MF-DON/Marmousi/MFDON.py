# %%
# !pip install sciann fteikpy pyDOE -q 

# %%
import time
import os
# %%
import numpy as np
import sciann as sn 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tensorflow import keras
from keras.layers import Input, Dense, concatenate, Multiply, Add, Subtract,Dot
from keras.models import Model
from keras.optimizers  import Adam 
#%matplotlib inline
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback,ModelCheckpoint
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
rcParams['font.family']='Times New Roman'
# %%
PATH_NAME = "DON/En-DeepONet/Lossplot/"
#PATH_NAME = "D:/deeponet/En-DeepONet-main/Marmousi/test/result/"

# %%

#tf.test.gpu_device_name()

# %%
MARMOUSI = pd.read_csv('DON/En-DeepONet/vel.rsf@', index_col=None, header=None)
VELMODEL = np.reshape(np.array(MARMOUSI), (2301, 751))[:, ::-1] #.T[::-1]

# %%
XMIN, XMAX = 0, 9.2
YMIN, YMAX = 0, 3

# %%
XGRID = np.linspace(XMIN, XMAX, 2301)
YGRID = np.linspace(YMIN, YMAX, 751)
XGRID, YGRID = np.meshgrid(XGRID, YGRID, indexing='ij')

#plt.figure(figsize= (10, 3))
#plt.colorbar(plt.pcolor(XGRID, YGRID, VELMODEL, cmap='brg'))
# plt.gca().invert_yaxis()
#plt.show()
#save_path = os.path.join(PATH_NAME, "velmodel.png")
#plt.savefig(save_path)

# %%
def interpolate_velocity_model(Xs, Ys, method='nearest'):
    crd = np.hstack([XGRID.reshape(-1,1), YGRID.reshape(-1,1)])
    Vs = griddata(crd, VELMODEL.flatten(), (Xs, Ys), method=method)
    return Vs.reshape(Xs.shape)

# %%
xt, yt = np.meshgrid(np.linspace(XMIN, XMAX, 100), np.linspace(YMIN, YMAX, 50), indexing='ij')
vt = interpolate_velocity_model(xt, yt, 'nearest')
font_tnr = FontProperties(family='Times New Roman')
title_font = {'fontproperties': font_tnr, 'size': 16}
label_font = {'fontproperties': font_tnr, 'size': 14}
annot_font = {'fontproperties': font_tnr, 'size': 14}

# 构建图像
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 子图1：原始速度模型
pcm1 = ax1.pcolor(XGRID, YGRID, VELMODEL, cmap='rainbow')
fig.colorbar(pcm1, ax=ax1)
ax1.set_title("Marmousi Velocity Model", **title_font)
ax1.set_xlabel("Offset (km)", **label_font)
ax1.set_ylabel("Depth (km)", **label_font)

# 子图2：插值速度模型
pcm2 = ax2.pcolor(xt, yt, vt, cmap='rainbow')
fig.colorbar(pcm2, ax=ax2)
ax2.set_title("Interpolated Velocity Model", **title_font)
ax2.set_xlabel("Offset (km)", **label_font)
ax2.set_ylabel("Depth (km)", **label_font)

# 添加 (a)、(b) 标注（居中放在图下方）
ax1.text(0.5, -0.2, "(a)", transform=ax1.transAxes, ha='center', va='top', **annot_font)
ax2.text(0.5, -0.2, "(b)", transform=ax2.transAxes, ha='center', va='top', **annot_font)

# 保存图像
plt.tight_layout()
save_path = os.path.join(PATH_NAME, "combined_velocity_models.png")
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
# %% [markdown]
# # Testing different velocity models 

# %%
import skfmm
from scipy.interpolate import griddata

class EikonalSolver:
    def __init__(self, xd = [-1.0, 1.0],
                       yd = [-1.0, 1.0],
                       vel = np.ones((10, 10)),
                       source = [0.0, 0.0]):
        self.origin = (xd[0], yd[0])
        Nx, Ny = [n - 1 for n in vel.shape]
        dx, dy = (xd[1]-xd[0])/(Nx), (yd[1]-yd[0])/(Ny)
        sx, sy = np.round((source[0] - xd[0])/dx).astype(int), np.round((source[1] - yd[0])/dy).astype(int)
        phi = np.ones_like(vel)
        phi[sx, sy] = -1
        self.nx = (Nx+1, Ny+1)
        self.xg = np.meshgrid(np.linspace(xd[0], xd[1], Nx+1), np.linspace(yd[0], yd[1], Ny+1), indexing='ij')
        self.dx = (dx, dy)
        self.vg = vel
        self.Tg = skfmm.travel_time(phi,vel,dx=(dx, dy),order=2)
        
    def __call__(self, xs=0., ys=0.):
        crd = np.hstack([self.xg[0].reshape(-1,1), self.xg[1].reshape(-1,1)])
        ts = griddata(crd, self.Tg.flatten(), (xs, ys), method='nearest')
        return ts.reshape(xs.shape)
    
    @property
    def grid(self):
        return self.Tg
    
    @property
    def grad(self):
        Tx, Ty = np.gradient(self.Tg)
        return (Tx/self.dx[0], Ty/self.dx[1])
  

# %%

for i in range(3):
    xs = np.random.uniform(XMIN, XMAX)
    ys = np.random.uniform(YMIN, YMAX)
    print("source =  ", xs, ys)
    eik = EikonalSolver(xd=[XMIN, XMAX], yd=[YMIN, YMAX], vel=VELMODEL, source = [xs, ys])
    t = eik.grid
    gt = eik.grad
    res = abs(gt[0]**2 + gt[1]**2 - 1/VELMODEL**2)
    
    Nx, Ny = 300, 100
    xp, yp = np.meshgrid(np.linspace(XMIN,XMAX,Nx), np.linspace(YMIN,YMAX,Ny), indexing='ij')
    tp = eik(xp, yp)

    fig, ax = plt.subplots(1,2,figsize=(15,3))
    # plt.colorbar(
    #     ax[0].pcolor(XGRID, YGRID, VELMODEL, cmap='seismic'), ax=ax[0]
    # )
    plt.colorbar(
        ax[0].contour(XGRID, YGRID, t, 10, cmap='seismic'), ax=ax[0]
    )
    plt.colorbar(
        ax[1].contour(XGRID, YGRID, t, 10, cmap='seismic'), ax=ax[1]
    )
    # plt.colorbar(
    #     ax[2].pcolor(xgrid, ygrid, res, cmap='seismic'), ax=ax[2]
    # )
    for j in range(len(ax)):
        ax[j].scatter(xs, ys, marker='*', color='r')
    #plt.show()
    save_path = os.path.join(PATH_NAME, f"plotSource_{i}.png")
    plt.savefig(save_path)

# %%
from pyDOE import *

def lhs_uniform(d, n, bb = None):
    samples = lhs(d, n)
    if not bb:
        bb = [[0, 1] for i in range(d)]
    data = []
    for i, bb_i, in zip(range(d), bb):
        data.append(
            bb_i[0] + (bb_i[1]-bb_i[0]) * samples[:, i]
        )
    return data if len(data)>1 else data[0]

# %%
SENSOR_SIZE = 20

# %%
keras.backend.clear_session()
tf.random.set_seed(1234)
# Define the model
class DeepONet(Model):
    def __init__(self, sensor_size=20, 
                 embedding_size=20,
                 trunk_layers=2*[50],
                 branch_layers=2*[50],
                 actf='tanh'):
        super(DeepONet, self).__init__()
        
        # 定义输入层
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')
        
        # 构建分支和主干网络（无特征交互）
        self.branch_net  = self._build_network(self.branch_input, branch_layers, actf, embedding_size)
        self.trunk_net  = self._build_network(
            concatenate([self.trunk_input_x, self.trunk_input_y]), 
            trunk_layers, actf, embedding_size 
        )
        
        # 直接拼接分支和主干的输出，跳过特征交互层和 Root 网络
        #merged = concatenate([self.trunk_net, self.branch_net])
        dot_product = Dot(axes=1)([self.trunk_net, self.branch_net])
        # 输出层
        self.output_layer  = dot_product
        # 显式定义模型输入输出
        super().__init__(
            inputs=[self.trunk_input_x, self.trunk_input_y, self.branch_input], 
            outputs=self.output_layer  
        )
    
    def _build_network(self, input_layer, layers, activation, final_units):
        x = input_layer 
        for units in layers:
            x = Dense(units, activation=activation)(x)
        return Dense(final_units, activation=activation)(x)
    
    def compile(self, **kwargs):
        # 初始化Eikonal损失记录器 
        self.eikonal_loss  = tf.keras.metrics.Mean(name='eikonal_loss') 
        super().compile(**kwargs)
        
    def train_step(self, data):
        # 解包数据（x_trunk, y_trunk为坐标，t_branch为传感器数据）
        (x_trunk, y_trunk, t_branch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(VELMODEL)) 
            
            # 总损失（数据+物理约束）
            #total_loss = data_loss + 0.1 * eikonal_loss 
            total_loss = data_loss
        # 梯度回传（手动释放persistent tape）
        gradients = tape.gradient(total_loss,  self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients,  self.trainable_variables)) 
        del tape  # 显式释放持久梯度带 
        
        # 更新监控指标 
        self.compiled_metrics.update_state(y_true,  y_pred)
        self.eikonal_loss.update_state(eikonal_loss) 
        
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "eikonal_loss": self.eikonal_loss.result(), 
            **{m.name:  m.result()  for m in self.metrics} 
        }
class CA_DeepONet(Model):
    def __init__(self, sensor_size=20, 
                 embedding_size=20,
                 trunk_layers=2*[50],
                 branch_layers=2*[50],
                 root_layers=7*[20],
                 actf='tanh'):
        super(CA_DeepONet, self).__init__()
        
        # 定义输入层（核心修正：显式绑定到父类）
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')
        
        # 构建网络分支 
        self.branch_net  = self._build_network(self.branch_input,  branch_layers, actf, embedding_size)
        self.trunk_net  = self._build_network(
            concatenate([self.trunk_input_x, self.trunk_input_y]), 
            trunk_layers, actf, embedding_size 
        )
        # 定义通道注意力层
        def channel_attention(input_feature, ratio=4):
            channel = input_feature.shape[-1]
            
            # 全局平均池化
            x = tf.reduce_mean(input_feature, axis=1, keepdims=True)  # [batch, 1, channels]
            
            # 全连接层生成权重
            x = Dense(channel//ratio, activation='relu')(x)  # 降维
            x = Dense(channel, activation='sigmoid')(x)      # 恢复维度
            
            # 应用注意力权重
            return Multiply()([input_feature, x])
        
        # 对主干和分支分别应用通道注意力
        #self.trunk_attn = channel_attention(self.trunk_net)
        self.branch_attn = channel_attention(self.branch_net)
        
        # ----------------- 特征交互与合并 -----------------
        # 合并加权特征与原始特征（残差连接）
        interaction = concatenate([ 
            self.branch_attn, 
            self.trunk_net, 
            self.branch_net
        ])
        
        # ----------------- Root网络构建 -----------------
        self.root = interaction
        for units in root_layers:
            self.root  = Dense(units, activation=actf)(self.root) 
        
        # 输出层（直接绑定到父类）
        self.output_layer  = Dense(1, activation='linear', name='u')(self.root) 
        
        # 显式定义模型输入输出（关键修正！）
        super().__init__(
            inputs=[self.trunk_input_x, self.trunk_input_y,  self.branch_input], 
            outputs=self.output_layer  
        )
    
    def _build_network(self, input_layer, layers, activation, final_units):
        x = input_layer 
        for units in layers:
            x = Dense(units, activation=activation)(x)
        return Dense(final_units, activation=activation)(x)
    
    def compile(self, **kwargs):
        # 初始化Eikonal损失记录器 
        self.eikonal_loss  = tf.keras.metrics.Mean(name='eikonal_loss') 
        super().compile(**kwargs)
        
    def train_step(self, data):
        # 解包数据（x_trunk, y_trunk为坐标，t_branch为传感器数据）
        (x_trunk, y_trunk, t_branch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(VELMODEL)) 
            
            # 总损失（数据+物理约束）
            #total_loss = data_loss + 0.1 * eikonal_loss 
            total_loss = data_loss
        # 梯度回传（手动释放persistent tape）
        gradients = tape.gradient(total_loss,  self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients,  self.trainable_variables)) 
        del tape  # 显式释放持久梯度带 
        
        # 更新监控指标 
        self.compiled_metrics.update_state(y_true,  y_pred)
        self.eikonal_loss.update_state(eikonal_loss) 
        
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "eikonal_loss": self.eikonal_loss.result(), 
            **{m.name:  m.result()  for m in self.metrics} 
        }
class Root_DeepONet(Model):
    def __init__(self, sensor_size=20, 
                 embedding_size=20,
                 trunk_layers=2*[50],
                 branch_layers=2*[50],
                 root_layers=7*[20],
                 actf='tanh'):
        super(Root_DeepONet, self).__init__()
        
        # 定义输入层（核心修正：显式绑定到父类）
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')
        
        # 构建网络分支 
        self.branch_net  = self._build_network(self.branch_input,  branch_layers, actf, embedding_size)
        self.trunk_net  = self._build_network(
            concatenate([self.trunk_input_x, self.trunk_input_y]), 
            trunk_layers, actf, embedding_size 
        )
        
        # 特征交互层 
        self.multiply  = Multiply()([self.trunk_net, self.branch_net]) 
        self.add  = Add()([self.trunk_net, self.branch_net]) 
        self.subtract  = Subtract()([self.trunk_net, self.branch_net]) 
        
        # Root网络构建 
        self.root  = concatenate([self.multiply, self.add,  self.subtract]) 
        for units in root_layers:
            self.root  = Dense(units, activation=actf)(self.root) 
        
        # 输出层（直接绑定到父类）
        self.output_layer  = Dense(1, activation='linear', name='u')(self.root) 
        
        # 显式定义模型输入输出（关键修正！）
        super().__init__(
            inputs=[self.trunk_input_x, self.trunk_input_y,  self.branch_input], 
            outputs=self.output_layer  
        )
    
    def _build_network(self, input_layer, layers, activation, final_units):
        x = input_layer 
        for units in layers:
            x = Dense(units, activation=activation)(x)
        return Dense(final_units, activation=activation)(x)
    
    def compile(self, **kwargs):
        # 初始化Eikonal损失记录器 
        self.eikonal_loss  = tf.keras.metrics.Mean(name='eikonal_loss') 
        super().compile(**kwargs)
        
    def train_step(self, data):
        # 解包数据（x_trunk, y_trunk为坐标，t_branch为传感器数据）
        (x_trunk, y_trunk, t_branch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(VELMODEL)) 
            
            # 总损失（数据+物理约束）
            #total_loss = data_loss + 0.1 * eikonal_loss 
            total_loss = data_loss
        # 梯度回传（手动释放persistent tape）
        gradients = tape.gradient(total_loss,  self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients,  self.trainable_variables)) 
        del tape  # 显式释放持久梯度带 
        
        # 更新监控指标 
        self.compiled_metrics.update_state(y_true,  y_pred)
        self.eikonal_loss.update_state(eikonal_loss) 
        
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "eikonal_loss": self.eikonal_loss.result(), 
            **{m.name:  m.result()  for m in self.metrics} 
        }   
class LossLogger(tf.keras.callbacks.Callback): 
    def __init__(self, 
                 log_dir=PATH_NAME,  # 直接绑定用户指定路径 
                 file_name="physics_informed_loss.csv", 
                 save_freq=1):
        super().__init__()
        # 路径安全处理（跨平台兼容）
        self.log_dir  = os.path.normpath(log_dir)   # 自动修正路径分隔符 
        self.log_path  = os.path.join(self.log_dir,  file_name)
        self.save_freq  = save_freq 
        
        # 动态创建目录（递归创建多级目录）
        os.makedirs(self.log_dir,  exist_ok=True)
        
        # 仅首次运行时写入表头 
        if not os.path.exists(self.log_path): 
            with open(self.log_path,  'w') as f:
                f.write("epoch,loss,data_loss,eikonal_loss,mse\n") 
 
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq  == 0:
            logs = logs or {}
            loss_val = float(logs.get('loss',  0.0))
            data_loss_val = float(logs.get('data_loss',  0.0))
            eikonal_loss_val = float(logs.get('eikonal_loss',  0.0))
            mse_val = float(logs.get('mse',  0.0))
        
        # 关键修复2：正确多行f-string语法 
            log_line = (
                f"{epoch + 1},"
                f"{loss_val:.8f},"   # 使用更高精度
                f"{data_loss_val:.8f},"
                f"{eikonal_loss_val:.8f},"
                f"{mse_val:.8f}\n"
            )
            with open(self.log_path,  'a') as f:
                f.write(log_line)

class PerformanceLogger(Callback):
    def __init__(self, 
                 log_dir=PATH_NAME,
                 file_name="test_metrics.csv"):
        super().__init__()
        self.log_path = os.path.join(log_dir, file_name)
        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 初始化文件头
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("sample_id,x_source,y_source,mse,mae\n")

    def on_test_end(self, logs=None):
        # 此方法会在模型测试时自动调用
        pass  # 我们将在手动测试循环中直接调用记录方法

    @staticmethod
    def record_metrics(log_path, sample_id, x_source, y_source, y_true, y_pred):
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # 写入文件
        with open(log_path, 'a') as f:
            f.write(f"{sample_id},{x_source:.4f},{y_source:.4f},{mse:.6f},{mae:.6f},{r2:.6f}\n")
# %%
from keras.utils import Sequence 


class DataGenerator(Sequence):
    def __init__(self, sample_size = 100, 
                       batch_size = dict(data=1000, domain=1000, bc=100),
                       sensor_size = 10,
                       shuffle = True,
                       seed=1234):
        # generate data
        self._sample_size = sample_size
        self._batch_size = batch_size
        self._sensor_size = sensor_size
        self._sensor = [np.linspace(XMIN, XMAX, sensor_size+2)[1:-1],  np.full(sensor_size, YMAX)]
        self._shuffle = shuffle
        self._epoch = 1
        self._time0 = time.time()
        self._time = time.time()
        self._set_data()
        
    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return self._sample_size

    def __getitem__(self, index):
        idx = self._sample_ids[index]
        return self.inputs[idx], self.targets[idx], self.sample_weights[idx]

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._sample_ids)
        epoch_time = time.time() - self._time
        total_time = time.time() - self._time0
        self._time = time.time()
        if self._epoch % 10 == 0:
            print(f"{self._epoch} - epoch-time: {epoch_time:.3f}s - total-time: {total_time:.3f}s ")
        self._epoch += 1

    def _set_data(self):
        num_sample, batch_size = self._sample_size, self._batch_size
        inputs, targets, sample_weights = [], [], []
        for i in range(num_sample):
            delx, dely = XMAX-XMIN, YMAX-YMIN
            x_source = np.random.uniform(XMIN + 0.1*delx, XMAX - 0.1*delx)
            y_source = np.random.uniform(YMIN + 0.1*dely, YMAX - 0.1*dely)
            inputs_i, targets_i, sample_weights_i = self.__generate_batch_data(x_source, y_source, batch_size)
            inputs.append(inputs_i)
            targets.append(targets_i)
            sample_weights.append(sample_weights_i)
            if (i+1) % 10 == 0:
                print(f"sample {i+1} is generated")
        # concat data
        self.inputs = inputs
        self.targets = targets
        self.sample_weights = sample_weights
        self._sample_ids = np.arange(num_sample)

    def __generate_batch_data(self, x_source, y_source, batch_size):
        counter = 0
        
        # eikonal (travel-time) solution
        x_grid, y_grid = XGRID, YGRID
        eik2d = EikonalSolver(xd=[XMIN, XMAX], yd=[YMIN, YMAX], vel=VELMODEL, source=[x_source, y_source])
        
        # sample data
        ids_data = np.random.choice(x_grid.size, batch_size['data'], replace=False)
        x_data, y_data = x_grid.flatten()[ids_data], y_grid.flatten()[ids_data]
        target_data = eik2d.grid.flatten()[ids_data]
        ids_data = np.arange(batch_size['data']) + counter
        counter += ids_data.size

        size_sample = counter
        
        # trunk inputs
        #trunk_input = np.stack([x_data.reshape(-1,1),y_data.reshape(-1,1)])   
        x_trunk = x_data.reshape(-1,  1)  # 形状 (N, 1)
        y_trunk = y_data.reshape(-1,  1)  # 形状 (N, 1) 
        # - BRANCH -
        # velocity-branch inputs
        x_sensor, y_sensor = self._sensor
        t_sensor = eik2d(x_sensor, y_sensor)
        t_branch = np.tile(t_sensor, (size_sample, 1))
        # inputs
        inputs = [x_trunk , y_trunk , t_branch]
        targets, sample_weights = [], []
        for idx, tg in zip([ids_data],
                           [target_data]):
            wi = np.zeros(size_sample)
            wi[idx] = size_sample / idx.size
            sample_weights.append(wi)
            ti = np.zeros((size_sample, ))
            if isinstance(tg, np.ndarray):
                ti[idx] = tg
            targets.append(ti.reshape(-1,1))
        with open("DON/En-DeepONet/y_true_values.txt", 'w') as f:
        # 写入文件头 
            f.write(f"#  Targets Data | Samples: {len(targets)} | Dimensions: {targets[0].shape}\n")
            f.write(f"#  Generated on 2025-03-10 14:16 | Precision: {6} decimal places\n")
        
        # 遍历每个样本 
            for sample_idx, target_array in enumerate(targets):
            # 格式化数值 
                formatted_values = [
                    f"{val:.{6}f}" 
                    for val in target_array.flatten() 
                ]
            # 构造行内容 
                line = f"Sample-{sample_idx:04d} | " + " ".join(formatted_values)
                f.write(line  + "\n")
        return inputs, targets, sample_weights
    
    
    def generate_test_data(self, Xs, Ys, Nx=500, Ny=200, sensor_size=10,noise=0.):
        counter = 0
        
        # eikonal solution
        eik2d = EikonalSolver(xd=[XMIN, XMAX], yd=[YMIN, YMAX], vel=VELMODEL, source=[Xs, Ys])
        
        x_grid, y_grid = np.meshgrid(np.linspace(XMIN,XMAX,Nx), np.linspace(YMIN,YMAX,Ny), indexing='ij')
        t_grid = eik2d(x_grid, y_grid)
        
        # sample domain
        x_trunk, y_trunk = x_grid.reshape(-1,1), y_grid.reshape(-1,1)
        target = t_grid.reshape(-1,1)
        size_sample = Nx*Ny
        
        # velocity-branch inputs
        x_sensor, y_sensor = self._sensor
        t_sensor = eik2d(x_sensor, y_sensor)
        if noise > 0.:
            t_sensor += np.random.normal(0, 1, t_sensor.shape)*(noise*np.std(t_sensor))
            t_sensor = np.maximum(0., t_sensor)

        t_branch = np.tile(t_sensor, (size_sample, 1))
        
        # inputs
        inputs = [x_trunk, y_trunk, t_branch]
        
        return inputs, t_sensor, target
    
    
class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    """记录训练历史，包括损失值和时间信息"""
    def __init__(self, model_name, log_dir=PATH_NAME):
        super().__init__()
        self.model_name = model_name
        self.log_dir = log_dir
        self.history = {
            'epoch': [],
            'loss': [], 
            'data_loss': [], 
            'eikonal_loss': [], 
            'mse': [],
            'epoch_time': [],
            'cumulative_time': []
        }
        self.epoch_start_time = None
        self.training_start_time = None
        
        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        """训练开始时记录开始时间"""
        self.training_start_time = time.time()
        self.history = {
            'epoch': [],
            'loss': [], 
            'data_loss': [], 
            'eikonal_loss': [], 
            'mse': [],
            'epoch_time': [],
            'cumulative_time': []
        }
        
    def on_epoch_begin(self, epoch, logs=None):
        """每个epoch开始时记录时间"""
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时记录历史"""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            cumulative_time = time.time() - self.training_start_time
            
            # 记录数据
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(logs.get('loss', 0.0))
            self.history['data_loss'].append(logs.get('data_loss', 0.0))
            self.history['eikonal_loss'].append(logs.get('eikonal_loss', 0.0))
            self.history['mse'].append(logs.get('mse', 0.0))
            self.history['epoch_time'].append(epoch_time)
            self.history['cumulative_time'].append(cumulative_time)
            
    def on_train_end(self, logs=None):
        """训练结束时保存历史到CSV文件"""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.log_dir, f'{self.model_name}_training_history.csv')
        df.to_csv(csv_path, index=False)
        #print(f"\n训练历史已保存到: {csv_path}")

class CombinedLogger(tf.keras.callbacks.Callback):
    """结合原有的LossLogger和新的TrainingHistoryCallback"""
    def __init__(self, model_name, log_dir=PATH_NAME, save_freq=1):
        super().__init__()
        self.model_name = model_name
        self.log_dir = log_dir
        self.save_freq = save_freq
        
        # 训练历史记录
        self.history = {
            'epoch': [],
            'loss': [], 
            'data_loss': [], 
            'eikonal_loss': [], 
            'mse': [],
            'epoch_time': [],
            'cumulative_time': []
        }
        self.epoch_start_time = None
        self.training_start_time = None
        
        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 原有的physics_informed_loss文件
        self.physics_log_path = os.path.join(self.log_dir, f"physics_informed_loss_{model_name}.csv")
        if not os.path.exists(self.physics_log_path):
            with open(self.physics_log_path, 'w') as f:
                f.write("epoch,loss,data_loss,eikonal_loss,mse\n")
                
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录时间
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            cumulative_time = time.time() - self.training_start_time
            
            # 记录到历史
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(logs.get('loss', 0.0))
            self.history['data_loss'].append(logs.get('data_loss', 0.0))
            self.history['eikonal_loss'].append(logs.get('eikonal_loss', 0.0))
            self.history['mse'].append(logs.get('mse', 0.0))
            self.history['epoch_time'].append(epoch_time)
            self.history['cumulative_time'].append(cumulative_time)
            
            # 保存到physics_informed_loss文件（保持原有格式）
            if (epoch + 1) % self.save_freq == 0:
                logs = logs or {}
                loss_val = float(logs.get('loss', 0.0))
                data_loss_val = float(logs.get('data_loss', 0.0))
                eikonal_loss_val = float(logs.get('eikonal_loss', 0.0))
                mse_val = float(logs.get('mse', 0.0))
                
                log_line = (
                    f"{epoch + 1},"
                    f"{loss_val:.8f},"
                    f"{data_loss_val:.8f},"
                    f"{eikonal_loss_val:.8f},"
                    f"{mse_val:.8f}\n"
                )
                with open(self.physics_log_path, 'a') as f:
                    f.write(log_line)
                    
    def on_train_end(self, logs=None):
        # 保存完整的训练历史
        df = pd.DataFrame(self.history)
        history_path = os.path.join(self.log_dir, f'{self.model_name}_training_history.csv')
        df.to_csv(history_path, index=False)
        #print(f"\n训练历史已保存到: {history_path}")
# %%
dg = DataGenerator(
    sample_size=200,
    batch_size={'data': 10000, 'domain': 1000, 'bc':50},
    sensor_size=SENSOR_SIZE
)

# %%
from itertools import cycle

n_plot = 4
fig, ax = plt.subplots(1, n_plot, figsize=(15, 3))
for i in range(n_plot):
    cycol = cycle('brycmg')
    inputs, targets, weights = dg[np.random.choice(len(dg))]
    for wi in weights:
        x_trunk, y_trunk, t_branch = inputs
        idx = wi > 0
        ax[i].scatter(x_trunk[idx], y_trunk[idx], color=next(cycol), alpha=0.2)
plt.savefig(os.path.join(PATH_NAME, 'nplot.png'))
#plt.show()

# %%
class ConditionedLRScheduler(Callback):
    def __init__(self, check_interval=100, patience=3, factor=0.1, min_lr=1e-6):
        super().__init__()
        self.check_interval = check_interval
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        # 仅在每100个epoch时检查
        if (epoch + 1) % self.check_interval != 0:
            return

        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # 使用 K.get_value 替代 .numpy()
                old_lr = K.get_value(self.model.optimizer.lr)
                new_lr = max(old_lr * self.factor, self.min_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch+1}: 损失未改善，学习率从 {old_lr:.2e} 衰减至 {new_lr:.2e}")
                self.wait = 0
CA_DON = CA_DeepONet(
    sensor_size=SENSOR_SIZE,
    embedding_size=100,
    branch_layers=[50],
    trunk_layers=[20, 50, 100],
    root_layers=[100, 80, 60, 40, 20],
    actf='tanh'
)
ROOT_DON = Root_DeepONet(
    sensor_size=SENSOR_SIZE,
    embedding_size=100,
    branch_layers=[50],
    trunk_layers=[20, 50, 100],
    root_layers=[100, 80, 60, 40, 20],
    actf='tanh'
)
DON = DeepONet(
    sensor_size=SENSOR_SIZE,
    embedding_size=100,
    branch_layers=[50],
    trunk_layers=[20, 50, 100],
    actf='tanh'
)
def lr_scheduler(epoch, lr):
    boundaries = [0, 500, 1500, 2000, 3000, 5000, 10000]
    values = [0.5e-4, 1e-4,2e-4, 0.75e-4, 0.5e-4, 1e-5, 1e-6]
    for i in range(len(boundaries)-1):
        if boundaries[i] <= epoch < boundaries[i+1]:
            return values[i]
    return values[-1]
optimizer = Adam(learning_rate=0.5e-4)
CA_DON.compile(optimizer=optimizer,loss="mae")
ROOT_DON.compile(optimizer=optimizer,loss="mae")
DON.compile(optimizer=optimizer,loss="mae")
SAVE_PERIOD = 100
checkpoint_CA = ModelCheckpoint(
    filepath=os.path.join(PATH_NAME, "weights_CA-epoch{epoch:06d}.hdf5"),
    save_weights_only=True,
    period=SAVE_PERIOD,
    verbose=1
)
combined_logger_CA = CombinedLogger(model_name="CA_DeepONet", log_dir=PATH_NAME)
checkpoint_ROOT = ModelCheckpoint(
    filepath=os.path.join(PATH_NAME, "weights_ROOT-epoch{epoch:06d}.hdf5"),
    save_weights_only=True,
    period=SAVE_PERIOD,
    verbose=1
)
combined_logger_ROOT = CombinedLogger(model_name="ROOT_DeepONet", log_dir=PATH_NAME)
checkpoint_DON = ModelCheckpoint(
    filepath=os.path.join(PATH_NAME, "weights_DON-epoch{epoch:06d}.hdf5"),
    save_weights_only=True,
    period=SAVE_PERIOD,
    verbose=1
)
combined_logger_DON = CombinedLogger(model_name="DeepONet", log_dir=PATH_NAME)
lr_callback = LearningRateScheduler(lr_scheduler)
EPOCHS = 5000
conditioned_lr = ConditionedLRScheduler(check_interval=100, patience=3, factor=0.5, min_lr=1e-6)
# 训练标志
TRAIN_MODELS = False  # 设置为True开始训练，False则加载已有权重

if TRAIN_MODELS:
    print("开始训练 CA-DeepONet...")
    start_time = time.time()
    
    CA_DON.fit(
        dg,
        epochs=EPOCHS,
        callbacks=[combined_logger_CA, lr_callback, conditioned_lr, checkpoint_CA],
        verbose=1
    )
    
    ca_train_time = time.time() - start_time
    print(f"CA-DeepONet 训练完成，总耗时: {ca_train_time:.2f} 秒")
    
    print("\n开始训练 ROOT-DeepONet...")
    start_time = time.time()
    
    ROOT_DON.fit(
        dg,
        epochs=EPOCHS,
        callbacks=[combined_logger_ROOT, lr_callback, conditioned_lr, checkpoint_ROOT],
        verbose=1
    )
    
    root_train_time = time.time() - start_time
    print(f"ROOT-DeepONet 训练完成，总耗时: {root_train_time:.2f} 秒")
    
    print("\n开始训练 DeepONet...")
    start_time = time.time()
    
    DON.fit(
        dg,
        epochs=EPOCHS,
        callbacks=[combined_logger_DON, lr_callback, conditioned_lr, checkpoint_DON],
        verbose=1
    )
    
    don_train_time = time.time() - start_time
    print(f"DeepONet 训练完成，总耗时: {don_train_time:.2f} 秒")
    
    # 保存总体训练时间对比
    time_comparison = pd.DataFrame({
        'Model': ['CA-DeepONet', 'ROOT-DeepONet', 'DeepONet'],
        'Total_Training_Time': [ca_train_time, root_train_time, don_train_time],
        'Epochs': [EPOCHS, EPOCHS, EPOCHS]
    })
    time_comparison.to_csv(os.path.join(PATH_NAME, 'training_time_comparison.csv'), index=False)
    
else:
    # 加载已有权重
    weight_path_CA = os.path.join(PATH_NAME, 'weights_CA-epoch005000.hdf5')
    CA_DON.load_weights(weight_path_CA)
    
    weight_path_ROOT = os.path.join(PATH_NAME, 'weights_ROOT-epoch005000.hdf5')
    ROOT_DON.load_weights(weight_path_ROOT)
    
    weight_path_DON = os.path.join(PATH_NAME, 'weights_DON-epoch005000.hdf5')
    DON.load_weights(weight_path_DON)
""" def cust_pcolor(AX, X, Y, P, Xs=None, Ys=None, title=None, cmap='rainbow', 
                xlabel=True, ylabel=False, vmin=None, vmax=None):
    ax1 = AX.pcolor(X, Y, P, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(ax1, ax=AX)
    if Xs and Ys:
        AX.scatter(Xs, Ys, marker='*', color='r')
    AX.set_title(title)
    AX.set_yticks(np.linspace(YMIN, YMAX, 7), np.linspace(YMAX, YMIN, 7))
    if xlabel: AX.set_xlabel('Offset (km)')
    if ylabel: AX.set_ylabel('Depth (km)') """
def cust_pcolor(AX, X, Y, P, Xs=None, Ys=None, title=None, cmap='rainbow',
                xlabel=False, ylabel=False, yticks=False, vmin=None, vmax=None):
    ax1 = AX.pcolor(X, Y, P, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(ax1, ax=AX)
    
    if Xs is not None and Ys is not None:
        AX.scatter(Xs, Ys, marker='*', color='r')

    if title:
        AX.set_title(title, fontdict={'family': 'Times New Roman', 'size': 24})

    if ylabel:
        AX.set_ylabel('Depth (km)', fontsize=12)

    if xlabel:
        AX.set_xlabel('Offset (km)', fontsize=12)

    # Y ticks 仅在最后一列保留
    if yticks:
        yticks_pos = np.linspace(Y.min(), Y.max(), 7)
        yticks_labels = [f"{label:.2f}" for label in np.linspace(Y.max(), Y.min(), 7)]
        AX.set_yticks(yticks_pos)
        AX.set_yticklabels(yticks_labels, fontsize=10)
    else:
        AX.set_yticks([])
        AX.set_yticklabels([])
Nx, Ny = 300, 100

np.random.seed(1234)

delx, dely = XMAX-XMIN, YMAX-YMIN

""" fig, ax = plt.subplots(3, 5, figsize=(18, 6))

for i, (xi, eta) in enumerate([(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]):
    
    x_source = XMIN + xi*delx
    y_source = YMIN + eta*dely
    
    test_data, sensor_input, target_data = dg.generate_test_data(
        x_source, y_source, Nx=Nx, Ny=Ny, sensor_size=SENSOR_SIZE)
    
    x_test = test_data[0].reshape(Nx, Ny)
    y_test = test_data[1].reshape(Nx, Ny)
    exact = target_data.reshape(Nx, Ny)
        
    pred = CA_DON.predict(test_data).reshape(Nx, Ny)
    predDON = ROOT_DON.predict(test_data).reshape(Nx, Ny)
    i_source, j_source = np.unravel_index(pred.argmin(), pred.shape)
    x_source_pred = x_test[i_source, j_source]
    y_source_pred = y_test[i_source, j_source]

    cust_pcolor(ax[i, 0], x_test, y_test, pred, x_source_pred, y_source_pred, 'MS-DeepONet: T(s)' if i==0 else None, 
                cmap='rainbow', vmin=exact.min(), vmax=exact.max(), ylabel=True, xlabel=True if i==2 else False)
    cust_pcolor(ax[i, 1], x_test, y_test, predDON, x_source, y_source, 
                'DON: T(s)' if i == 0 else None,
                cmap='rainbow', vmin=exact.min(), vmax=exact.max(),
                xlabel=(i == 2))
    cust_pcolor(ax[i, 2], x_test, y_test, exact, x_source, y_source, 'FNN: T(s)' if i==0 else None,
                'rainbow', xlabel=True if i==2 else False)
    
    error = abs(pred - exact) / exact.max() * 100
    cust_pcolor(ax[i, 3], x_test, y_test, error, title='|MS-DON Error| (%)' if i==0 else None,
                cmap='rainbow', xlabel=True if i==2 else False)
    error2 = abs(predDON - exact) / exact.max() * 100
    cust_pcolor(ax[i, 4], x_test, y_test, error2, title='|DON Error| (%)' if i==0 else None,
                cmap='rainbow', xlabel=True if i==2 else False)

plt.subplots_adjust(0.05, 0.08, 0.98, 0.94, 0.1, 0.3)
#print("saving to: ", os.path.join(PATH_NAME, 'preds-pub.png'))
plt.savefig(os.path.join(PATH_NAME, 'preds-pub.png'))
#plt.show() """

Nx, Ny = 300, 100

rmse_error = []
rmse_errorDON = []
rmse_errorRoot = []
mae_error = []      # 新增：存储 MS-DeepONet 的 MAE
mae_errorDON = []   # 新增：存储 DON 的 MAE
mae_errorRoot = []
r2_error = []       # 新增：存储 MS-DeepONet 的 R²
r2_errorDON = []    # 新增：存储 DON 的 R²
r2_errorRoot =[]
source_loc = []
source_error = []

np.random.seed(652573)

NUM_TRIAL = 500
num = 0
metrics = {
    "trial": [],       # 试验编号
    "x_source": [],    # 真实源 X 坐标
    "y_source": [],    # 真实源 Y 坐标
    "mse_msdon": [],   # MS-DeepONet 的 MSE
    "mse_don": [],     # DON 的 MSE
    "mse_root": [],
    "mae_msdon": [],   # MS-DeepONet 的 MAE
    "mae_don": [],     # DON 的 MAE
    "mae_root": [],
    "r2_msdon": [],    # MS-DeepONet 的 R²
    "r2_don": [],      # DON 的 R²
    "r2_root": [],
}
fig, ax = plt.subplots(5, 5, figsize=(22, 10))
for i in range(NUM_TRIAL):
    
    delx, dely = XMAX-XMIN, YMAX-YMIN
    x_source = np.random.uniform(XMIN + 0.1*delx, XMAX - 0.1*delx)
    y_source = np.random.uniform(YMIN + 0.1*dely, YMAX - 0.1*dely)
    
    test_data, sensor_input, target_data = dg.generate_test_data(
        x_source, y_source, Nx=Nx, Ny=Ny, sensor_size=SENSOR_SIZE)
    
    x_test = test_data[0].reshape(Nx, Ny)
    y_test = test_data[1].reshape(Nx, Ny)
    exact = target_data.reshape(Nx, Ny)
    predDON = DON.predict(test_data).reshape(Nx, Ny)  
    pred = CA_DON.predict(test_data).reshape(Nx, Ny)
    predRoot = ROOT_DON.predict(test_data).reshape(Nx, Ny)
    i_source, j_source = np.unravel_index(pred.argmin(), pred.shape)
    x_source_pred = x_test[i_source, j_source]
    y_source_pred = y_test[i_source, j_source]
    i_source_DON, j_source_DON = np.unravel_index(predDON.argmin(), predDON.shape)
    x_source_pred_DON = x_test[i_source_DON, j_source_DON]
    y_source_pred_DON = y_test[i_source_DON, j_source_DON]
    i_source_Root, j_source_Root = np.unravel_index(predRoot.argmin(), predRoot.shape)
    x_source_pred_Root = x_test[i_source_Root, j_source_Root]
    y_source_pred_Root = y_test[i_source_Root, j_source_Root]
    rmse_error.append(
        np.sqrt(np.mean(exact - pred)**2)
    )
    rmse_errorDON.append(
        np.sqrt(np.mean(exact - predDON)**2)
    )
    rmse_errorRoot.append(
        np.sqrt(np.mean(exact - predRoot)**2)
    )
    # 计算 MAE
    mae = np.mean(np.abs(exact - pred))
    maeDON = np.mean(np.abs(exact - predDON))
    maeRoot = np.mean(np.abs(exact - predRoot))
    mae_error.append(mae)
    mae_errorDON.append(maeDON)
    mae_errorRoot.append(maeRoot)
    # 计算 R²（决定系数）
    ss_total = np.sum((exact - np.mean(exact))**2)
    ss_res = np.sum((exact - pred)**2)
    ss_resDON = np.sum((exact - predDON)**2)
    ss_resRoot = np.sum((exact - x_source_pred_Root)**2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0
    r2DON = 1 - (ss_resDON / ss_total) if ss_total != 0 else 0
    r2Root = 1 - (ss_resRoot / ss_total) if ss_total != 0 else 0
    r2_error.append(r2)
    r2_errorDON.append(r2DON)
    r2_errorRoot.append(r2Root)
    source_loc.append(
        [x_source, y_source]
    )
    source_error.append(
        [x_source-x_source_pred, y_source-y_source_pred]
    )
    metrics["trial"].append(i)
    metrics["x_source"].append(x_source)
    metrics["y_source"].append(y_source)
    metrics["mse_msdon"].append(np.sqrt(np.mean(exact - pred)**2))
    metrics["mse_don"].append(np.sqrt(np.mean(exact - predDON)**2))
    metrics["mse_root"].append(np.sqrt(np.mean(exact - predRoot)**2))
    metrics["mae_msdon"].append(maeDON)
    metrics["mae_don"].append(mae)
    metrics["mae_root"].append(maeRoot)
    metrics["r2_msdon"].append(r2DON)
    metrics["r2_don"].append(r2)
    metrics["r2_root"].append(r2Root)
    if i % int(NUM_TRIAL/5) == 0:
        error_max = max(
            abs(pred - exact).max(), 
            abs(predDON - exact).max()
        )
        cust_pcolor(ax[num,0], x_test, y_test, exact, x_source, y_source, 'FNN: T(s)' if num==0 else None, 'rainbow',xlabel=True if num==4 else False,ylabel= True,yticks = False)
        cust_pcolor(ax[num,1], x_test, y_test, pred, x_source_pred, y_source_pred, 'MF-DeepONet: T(s) OURS' if num==0 else None, 'rainbow',
                    vmin=exact.min(), vmax=exact.max(),xlabel=True if num==4 else False, ylabel=False,yticks = False)
        cust_pcolor(ax[num,2], x_test, y_test, predDON, x_source_pred_DON, y_source_pred_DON, 
                    'DON: T(s)' if num==0 else None, 'rainbow', vmin=exact.min(), vmax=exact.max(),xlabel=True if num==4 else False, ylabel=False,yticks = False)
        cust_pcolor(ax[num,3], x_test, y_test, abs(pred - exact), title='|MF-DON Error| OURS' if num==0 else None, cmap='rainbow',vmax=error_max,xlabel=True if num==4 else False, ylabel=False,yticks = False)
        cust_pcolor(ax[num,4], x_test, y_test, abs(predDON - exact), title='|DON Error|' if num==0 else None, cmap='rainbow',vmax=error_max,xlabel=True if num==4 else False, ylabel=False,yticks = True)
        num=num+1
font_tnr = FontProperties(family='Times New Roman', size=12)
df_metrics = pd.DataFrame(metrics)
csv_path = os.path.join(PATH_NAME, "model_metrics.csv")
df_metrics.to_csv(csv_path, index=False)
plt.subplots_adjust(0.05, 0.08, 0.98, 0.94, 0.1, 0.3)
plt.savefig(os.path.join(PATH_NAME, f'preds.png'))
        #plt.show()
rmse_error = np.array(rmse_error)
rmse_errorDON = np.array(rmse_errorRoot)
weights = np.ones(len(rmse_error)) / len(rmse_error) * 100
weightsDON = np.ones(len(rmse_errorDON)) / len(rmse_errorDON) * 100
bins = np.linspace(0, 0.1, 20)
plt.figure(figsize=(4,3))
plt.hist(rmse_errorDON, bins=bins,weights=weights, 
         alpha=0.5, color='blue', label='MF-DON')
# 绘制第二个直方图
plt.hist(rmse_error,bins=bins,weights=weightsDON, 
         alpha=0.5, color='red', label='DON')
plt.xlabel('RMSE', fontproperties=font_tnr)
plt.ylabel('Percentage / %', fontproperties=font_tnr)
plt.xlim(0., 0.1)
plt.legend(prop=font_tnr)
plt.tight_layout()
plt.savefig(
    os.path.join(PATH_NAME, 'ekional_source_error_distribution.png')
)
mae_error = np.array(mae_error)
mae_errorDON = np.array(mae_errorDON)
weights = np.ones(len(mae_error)) / len(mae_error) * 100
weightsDON = np.ones(len(mae_errorDON)) / len(mae_errorDON) * 100
bins = np.linspace(0, 0.1, 20)
plt.figure(figsize=(4,3))
plt.hist(mae_error,bins=bins, weights=weights, 
         alpha=0.5, color='blue', label='MF-DON')
# 绘制第二个直方图
plt.hist(mae_errorDON,bins=bins,weights=weightsDON, 
         alpha=0.5, color='red', label='DON')
plt.xlabel('MAE', fontproperties=font_tnr)
plt.ylabel('Percentage / %', fontproperties=font_tnr)
plt.xlim(0., 0.1)
plt.legend(prop=font_tnr)
plt.tight_layout()
plt.savefig(
    os.path.join(PATH_NAME, 'mae.png')
)
plt.close()  # 关闭当前 figure 释放内存
#noise
Nx, Ny = 300, 100
NUM_TRIAL = 500

noise_vals =  [0.01, 0.1, 0.2, 0.5]
noise_error = []
noise_source_error = []
noise_errorDON = []
noise_source_errorDON = []

for noise in noise_vals:
    rmse_error2 = []
    rmse_error2DON =[]
    source_loc2 = []
    source_error2 = []
    source_error2DON = []

    for i in range(NUM_TRIAL):

        x_source = np.random.uniform(XMIN, XMAX)
        y_source = np.random.uniform(YMIN, YMAX)

        test_data, sensor_input, target_data = dg.generate_test_data(
            x_source, y_source, Nx=Nx, Ny=Ny, sensor_size=SENSOR_SIZE, noise=noise)

        x_test = test_data[0].reshape(Nx, Ny)
        y_test = test_data[1].reshape(Nx, Ny)
        exact = target_data.reshape(Nx, Ny)

        pred = CA_DON.predict(test_data).reshape(Nx, Ny)
        predDON = DON.predict(test_data).reshape(Nx, Ny)
        i_source, j_source = np.unravel_index(pred.argmin(), pred.shape)
        i_sourceDON, j_sourceDON = np.unravel_index(predDON.argmin(), predDON.shape)
        x_source_pred = x_test[i_source, j_source]
        y_source_pred = y_test[i_source, j_source]
        x_source_predDON = x_test[i_sourceDON, j_sourceDON]
        y_source_predDON = y_test[i_sourceDON, j_sourceDON]
        rmse_error2.append(
            np.sqrt(np.mean(exact - pred)**2)
        )
        rmse_error2DON.append(
            np.sqrt(np.mean(exact - predDON)**2)
        )
        source_loc2.append(
            [x_source, y_source]
        )
        source_error2.append(
            [x_source-x_source_pred, y_source-y_source_pred]
        )
        source_error2DON.append(
            [x_source-x_source_predDON, y_source-y_source_predDON]
        )
    noise_error.append(rmse_error2)
    noise_source_error.append(source_error2)
    noise_errorDON.append(rmse_error2DON)
    noise_source_errorDON.append(source_error2DON)
noise_source_error = np.array(noise_source_error)
noise_source_errorDON = np.array(noise_source_errorDON)
noise_error = np.array(noise_error)
noise_errorDON = np.array(noise_errorDON)  # 确保变量名一致（原代码有拼写错误）

plt.figure(figsize=(4,3))

# 绘制 MSDON 数据（原模型）
plt.loglog(noise_vals, np.mean(noise_error, axis=-1), 
         '--*', color='blue', label='MF-DON')

# 添加 DON 模型数据（新增部分）
plt.loglog(noise_vals, np.mean(noise_errorDON, axis=-1), 
         '--o', color='red', label='DON')

plt.xlabel('Noise Ratio', fontproperties=font_tnr)
plt.ylabel('$\mathbb{E}$(RMSE)', fontproperties=font_tnr)
plt.legend(prop=font_tnr)  # 添加图例
plt.tight_layout()
plt.savefig(os.path.join(PATH_NAME, 'ekional_source_noise_sensitivity.png'))
def plot_training_comparison():
    """绘制训练历史比较图"""
    
    # 读取训练历史文件
    history_files = {
        'MF-DeepONet': os.path.join(PATH_NAME, 'CA_DeepONet_training_history.csv'),
        'En-DeepONet': os.path.join(PATH_NAME, 'ROOT_DeepONet_training_history.csv'),
        'DeepONet': os.path.join(PATH_NAME, 'DeepONet_training_history.csv')
    }
    
    # 检查文件是否存在
    existing_files = {name: path for name, path in history_files.items() if os.path.exists(path)}
    
    if not existing_files:
        print("未找到训练历史文件，请先训练模型")
        return
    
    # 设置字体
    font_tnr = FontProperties(family='Times New Roman')
    title_font = {'fontproperties': font_tnr, 'size': 16}
    label_font = {'fontproperties': font_tnr, 'size': 14}
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, file_path) in enumerate(existing_files.items()):
        df = pd.read_csv(file_path)
        color = colors[i % len(colors)]
        
        # 归一化损失
        normalized_loss = df['loss'] / df['loss'].iloc[0]
        
        # 1. 损失 vs 轮次
        axes[0, 0].semilogy(df['epoch'], normalized_loss, 
                           alpha=0.75, label=model_name, linewidth=2, color=color)
        
        # 2. 损失 vs 时间
        axes[0, 1].semilogy(df['cumulative_time'], normalized_loss, 
                           alpha=0.75, label=model_name, linewidth=2, color=color)
        
        # 3. 每轮次训练时间
        axes[0, 2].plot(df['epoch'], df['epoch_time'], 
                       alpha=0.75, label=model_name, linewidth=2, color=color)
        
        # 4. 数据损失
        axes[1, 0].semilogy(df['epoch'], df['data_loss'], 
                           alpha=0.75, label=model_name, linewidth=2, color=color)
        
        # 5. Eikonal损失
        axes[1, 1].semilogy(df['epoch'], df['eikonal_loss'], 
                           alpha=0.75, label=model_name, linewidth=2, color=color)
        
        # 6. MSE
        axes[1, 2].semilogy(df['epoch'], df['mse'], 
                           alpha=0.75, label=model_name, linewidth=2, color=color)
    
    # 设置标签和标题
    titles = [
        'Normalized Loss vs Epochs',
        'Normalized Loss vs Training Time',
        'Training Time per Epoch',
        'Data Loss vs Epochs',
        'Eikonal Loss vs Epochs',
        'MSE vs Epochs'
    ]
    
    xlabels = [
        'Epochs', 'Training Time (s)', 'Epochs',
        'Epochs', 'Epochs', 'Epochs'
    ]
    
    ylabels = [
        'Normalized Loss ($\mathcal{L}/\mathcal{L}_0$)',
        'Normalized Loss ($\mathcal{L}/\mathcal{L}_0$)',
        'Time per Epoch (s)',
        'Data Loss',
        'Eikonal Loss',
        'MSE'
    ]
    
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], **title_font)
        ax.set_xlabel(xlabels[i], **label_font)
        ax.set_ylabel(ylabels[i], **label_font)
        ax.legend(prop=font_tnr)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_NAME, 'training_comparison.png'), 
                dpi=300, bbox_inches='tight')
    #plt.show()

def plot_training_comparison():
    """绘制训练历史比较图 - 仅显示归一化损失对比"""
    
    # 读取训练历史文件
    history_files = {
        'MF-DeepONet': os.path.join(PATH_NAME, 'CA_DeepONet_training_history.csv'),
        'En-DeepONet': os.path.join(PATH_NAME, 'ROOT_DeepONet_training_history.csv'),
        'DeepONet': os.path.join(PATH_NAME, 'DeepONet_training_history.csv')
    }
    
    # 检查文件是否存在
    existing_files = {name: path for name, path in history_files.items() if os.path.exists(path)}
    
    if not existing_files:
        print("未找到训练历史文件，请先训练模型")
        return
    
    # 设置字体
    font_tnr = FontProperties(family='Times New Roman')
    title_font = {'fontproperties': font_tnr, 'size': 18}
    label_font = {'fontproperties': font_tnr, 'size': 16}
    legend_font = {'family': 'Times New Roman', 'size': 14}
    
    # 创建子图 - 只有2个图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    linestyles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    for i, (model_name, file_path) in enumerate(existing_files.items()):
        df = pd.read_csv(file_path)
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        # 归一化损失
        normalized_loss = df['loss'] / df['loss'].iloc[0]
        
        # 为了更好的可视化效果，可以对数据进行采样
        # 每10个epoch取一个点用于显示标记
        sample_indices = range(0, len(df), max(1, len(df)//50))
        
        # 1. 损失 vs 轮次
        axes[0].semilogy(df['epoch'], normalized_loss, 
                        alpha=0.8, label=model_name, linewidth=2.5, 
                        color=color, linestyle=linestyle)
        # 添加标记点（稀疏显示）
        axes[0].semilogy(df['epoch'].iloc[sample_indices], 
                        normalized_loss.iloc[sample_indices],
                        marker=marker, color=color, markersize=4, 
                        markevery=1, alpha=0.7)
        
        # 2. 损失 vs 时间
        axes[1].semilogy(df['cumulative_time'], normalized_loss, 
                        alpha=0.8, label=model_name, linewidth=2.5, 
                        color=color, linestyle=linestyle)
        # 添加标记点（稀疏显示）
        axes[1].semilogy(df['cumulative_time'].iloc[sample_indices], 
                        normalized_loss.iloc[sample_indices],
                        marker=marker, color=color, markersize=4, 
                        markevery=1, alpha=0.7)
    
    # 设置标签和标题
    titles = [
        'Normalized Loss vs Epochs',
        'Normalized Loss vs Training Time'
    ]
    
    xlabels = [
        'Epochs', 
        'Training Time (s)'
    ]
    
    ylabels = [
        'Normalized Loss ($\\mathcal{L}/\\mathcal{L}_0$)',
        'Normalized Loss ($\\mathcal{L}/\\mathcal{L}_0$)'
    ]
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], **title_font, pad=20)
        ax.set_xlabel(xlabels[i], **label_font)
        ax.set_ylabel(ylabels[i], **label_font)
        ax.legend(prop=legend_font, frameon=True, fancybox=True, 
                 shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # 设置坐标轴刻度字体
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 添加子图标注 (a), (b)
        ax.text(0.02, 0.98, f'({chr(97+i)})', transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(PATH_NAME, 'training_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 同时保存为PDF格式（适合论文使用）
    pdf_path = os.path.join(PATH_NAME, 'training_comparison.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"训练对比图已保存到: {save_path}")
    print(f"PDF版本已保存到: {pdf_path}")
    
    # 显示图像（如果需要的话，可以注释掉）
    # plt.show()
    
    plt.close()  # 关闭图像以释放内存

def generate_comparison_report():
    """生成详细的比较报告"""
    
    history_files = {
        'MF-DeepONet': os.path.join(PATH_NAME, 'CA_DeepONet_training_history.csv'),
        'En-DeepONet': os.path.join(PATH_NAME, 'ROOT_DeepONet_training_history.csv'),
        'DeepONet': os.path.join(PATH_NAME, 'DeepONet_training_history.csv')
    }
    
    existing_files = {name: path for name, path in history_files.items() if os.path.exists(path)}
    
    if not existing_files:
        print("未找到训练历史文件，请先训练模型")
        return
    
    comparison_data = []
    
    for model_name, file_path in existing_files.items():
        df = pd.read_csv(file_path)
        
        total_time = df['cumulative_time'].iloc[-1]
        avg_time_per_epoch = df['epoch_time'].mean()
        final_loss = df['loss'].iloc[-1]
        initial_loss = df['loss'].iloc[0]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        # 找到达到特定损失阈值的轮次
        target_loss = initial_loss * 0.1  # 10%的初始损失
        converged_epoch = None
        converged_time = None
        
        for i, loss in enumerate(df['loss']):
            if loss <= target_loss:
                converged_epoch = df['epoch'].iloc[i]
                converged_time = df['cumulative_time'].iloc[i]
                break
        
        # 计算不同阶段的收敛速度
        # 前25%轮次的损失降低率
        quarter_idx = len(df) // 4
        quarter_loss_reduction = (initial_loss - df['loss'].iloc[quarter_idx]) / initial_loss * 100
        
        # 前50%轮次的损失降低率  
        half_idx = len(df) // 2
        half_loss_reduction = (initial_loss - df['loss'].iloc[half_idx]) / initial_loss * 100
        
        print(f"\n{model_name}:")
        print(f"  总训练时间: {total_time:.2f} 秒")
        print(f"  平均每轮时间: {avg_time_per_epoch:.3f} 秒")
        print(f"  初始损失: {initial_loss:.6f}")
        print(f"  最终损失: {final_loss:.6f}")
        print(f"  总损失降低: {loss_reduction:.2f}%")
        print(f"  前25%轮次损失降低: {quarter_loss_reduction:.2f}%")
        print(f"  前50%轮次损失降低: {half_loss_reduction:.2f}%")
        
        if converged_epoch is not None:
            print(f"  达到10%初始损失的轮次: {converged_epoch}")
            print(f"  达到收敛的时间: {converged_time:.2f} 秒")
            convergence_efficiency = converged_time / total_time * 100
            print(f"  收敛效率: {convergence_efficiency:.1f}% (收敛时间/总时间)")
        else:
            print(f"  未达到10%初始损失阈值")
            
        # 收集数据用于CSV报告
        comparison_data.append({
            'Model': model_name,
            'Total_Time_s': total_time,
            'Avg_Time_Per_Epoch_s': avg_time_per_epoch,
            'Initial_Loss': initial_loss,
            'Final_Loss': final_loss,
            'Total_Loss_Reduction_%': loss_reduction,
            'Quarter_Loss_Reduction_%': quarter_loss_reduction,
            'Half_Loss_Reduction_%': half_loss_reduction,
            'Converged_Epoch': converged_epoch if converged_epoch else 'Not Converged',
            'Converged_Time_s': converged_time if converged_time else 'Not Converged',
            'Convergence_Efficiency_%': (converged_time / total_time * 100) if converged_time else 'Not Converged'
        })
    
    # 保存比较报告到CSV
    comparison_df = pd.DataFrame(comparison_data)
    report_path = os.path.join(PATH_NAME, 'model_comparison_report.csv')
    comparison_df.to_csv(report_path, index=False)
    print(f"\n详细比较报告已保存到: {report_path}")
    
    # 生成简化的性能总结表
    summary_data = []
    for _, row in comparison_df.iterrows():
        summary_data.append({
            'Model': row['Model'],
            'Training_Time_min': f"{row['Total_Time_s']/60:.1f}",
            'Final_Loss': f"{row['Final_Loss']:.2e}",
            'Loss_Reduction_%': f"{row['Total_Loss_Reduction_%']:.1f}",
            'Convergence_Status': 'Converged' if row['Converged_Epoch'] != 'Not Converged' else 'Not Converged'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(PATH_NAME, 'performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"性能总结表已保存到: {summary_path}")

# 在训练完成后调用比较函数
if TRAIN_MODELS:
    print("\n生成训练比较图表...")
    plot_training_comparison()
    generate_comparison_report()
else:
    # 如果没有训练，但想生成图表（假设已有历史文件）
    print("生成训练比较图表...")
    plot_training_comparison()
    generate_comparison_report()

