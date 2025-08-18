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
from collections import OrderedDict
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
rcParams['font.family']='Times New Roman'
# %%
#PATH_NAME = "DON/En-DeepONet/OpenFWI/test"
#PATH_NAME = "D:/deeponet/En-DeepONet-main/Marmousi/test/result/"
PATH_NAME = "DON/En-DeepONet/OpenFWI/testFault"

# xin
RECEIVER_SIZE = 30
SENSOR_SIZE = 14*14
SOURCE_RADI = 0.
VMIN = 0.2
VMAX = 8.0
EPOCHS_MAX = 20000
#tf.test.gpu_device_name()


vel_models_path = "DON/En-DeepONet/OpenFWI/CurveFault_Btest/"
#vel_models_path = "DON/En-DeepONet/OpenFWI/CurveFault_B/"
vel_models_list = list(filter(lambda x: x.startswith('vel'), sorted(os.listdir(vel_models_path))))
vel_models = np.load(os.path.join(vel_models_path, vel_models_list[0])) / 1000

INDEXING = 'ij'
DL = 0.050 #km
XMIN, XMAX = 0, 70*DL
YMIN, YMAX = 0, 70*DL
DELTAX = XMAX - XMIN
DELTAY = YMAX - YMIN
# %%
XGRID, YGRID = np.meshgrid(np.linspace(XMIN, XMAX, 70),
                           np.linspace(YMIN, YMAX, 70),
                           indexing=INDEXING)

def interpolate_velocity_model(Vs, Xs, Ys, method='nearest'):
    crd = np.hstack([XGRID.reshape(-1,1), YGRID.reshape(-1,1)])
    Vs = griddata(crd, Vs.flatten(), (Xs, Ys), method=method)
    return Vs.reshape(Xs.shape)
# %%
xgrid, ygrid = np.meshgrid(np.linspace(XMIN, XMAX, 70),
                           np.linspace(YMIN, YMAX, 70),
                           indexing=INDEXING)

vgrid = interpolate_velocity_model(vel_models[0, 0], xgrid, ygrid, 'nearest') #[:, ::-1]

fig, ax = plt.subplots(1, 4, figsize=(20, 4))  # 宽度调整为 20 英寸
model_indices = [100, 200, 300, 400]
for i, idx in enumerate(model_indices):
    # 绘制伪彩色图
    pc = ax[i].pcolor(XGRID, YGRID, vel_models[idx, 0], cmap='rainbow')
    # 添加颜色条（缩进比例 80% 避免过大）
    plt.colorbar(pc, ax=ax[i], shrink=0.8)
    # 设置标题
    ax[i].set_title(f"Model {idx}", fontdict={'family': 'Times New Roman', 'size': 16})
plt.tight_layout(w_pad=3.0)  # 水平间距 3 单位
save_path = os.path.join(PATH_NAME, "multi_velmodels.png")
plt.savefig(save_path, bbox_inches='tight', dpi=150)  # 提升保存质量
""" fig, ax = plt.subplots(1,2,figsize=(8, 3))
ax[0].scatter(xgrid.flatten(), ygrid.flatten(), c=vel_models[0, 0].flatten(), cmap='jet')
# ax[0].invert_yaxis()
plt.colorbar(
    ax[1].pcolor(XGRID, YGRID, vel_models[0, 0], cmap='jet')
)
save_path = os.path.join(PATH_NAME, f"velmodel.png")
plt.savefig(save_path) """

xgrid, ygrid = np.meshgrid(np.linspace(XMIN, XMAX, 100),
                           np.linspace(YMIN, YMAX, 100),
                           indexing=INDEXING)

vgrid = interpolate_velocity_model(vel_models[0, 0], xgrid, ygrid, 'nearest')

fig, ax = plt.subplots(1,2,figsize=(8, 3))
ax[0].pcolor(xgrid, ygrid, vgrid, cmap='seismic')
# ax[0].invert_yaxis()
ax[1].pcolor(XGRID, YGRID, vel_models[0, 0], cmap='seismic')
save_path = os.path.join(PATH_NAME, f"velmodelinterpolate.png")
plt.savefig(save_path)

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
    v = vel_models[i, 0]
    xs = np.random.uniform(XMIN, XMAX)
    ys = np.random.uniform(YMIN, YMAX)
    #print("source =  ", xs, ys)
    eik = EikonalSolver(xd=[XMIN, XMAX], yd=[YMIN, YMAX], vel=v, source = [xs, ys])
    t = eik.grid
    x, y = eik.xg
    gt = eik.grad
    res = abs(gt[0]**2 + gt[1]**2 - 1/v**2)
    
    fig, ax = plt.subplots(1,3,figsize=(12,3))
    plt.colorbar(
        ax[0].pcolor(x, y, v.T, cmap='seismic'), ax=ax[0]
    )
    ax[0].plot(ys, xs, '*w')
    plt.colorbar(
        ax[1].contour(x, y, t.T, cmap='seismic'), ax=ax[1]
    )
    ax[1].plot(ys, xs, '*r')
    plt.colorbar(
        ax[2].pcolor(x, y, res.T, cmap='seismic'), ax=ax[2]
    )
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
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
keras.backend.clear_session()
tf.random.set_seed(1234)
# Define the model
class DeepONet(Model):
    def __init__(self, sensor_size=20, 
                 receiver_size=20,
                 embedding_size=100,
                 trunk_layers=[50],
                 sensor_branch_layers=7*[20],
                 receiver_branch_layers=7*[20],
                 actf='tanh'):
        super(DeepONet, self).__init__()
        
        # 定义输入层（核心修正：显式绑定到父类）
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.r_branch_inputs  = Input(shape=(receiver_size,), name='rb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')

        self.sensor_branch_net  = self._build_network(self.branch_input, sensor_branch_layers, actf, embedding_size)
        self.receiver_branch_net  = self._build_network(self.r_branch_inputs, receiver_branch_layers, actf, embedding_size)
        self.trunk_net  = self._build_network(
            concatenate([self.trunk_input_x, self.trunk_input_y]), 
            trunk_layers, actf, embedding_size 
        )
        
        # 分别计算两个点积
        sensor_dot = Dot(axes=1)([self.trunk_net, self.sensor_branch_net])
        receiver_dot = Dot(axes=1)([self.trunk_net, self.receiver_branch_net])  # 第二个点积
        
        # 合并点积结果（加法操作）
        combined_output = Add()([sensor_dot, receiver_dot])
        
        # 定义最终输出
        self.output_layer = combined_output
        # 显式定义模型输入输出（关键修正！）
        super().__init__(
            inputs=[
                self.trunk_input_x, 
                self.trunk_input_y, 
                self.branch_input,
                self.r_branch_inputs  # 新增输入项
            ],
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
        (x_trunk, y_trunk, t_branch , r_rbranch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch , r_rbranch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(v)) 
            
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
    def __init__(self, receiver_size=20,
                 sensor_size=20, 
                 embedding_size=100,
                 trunk_layers=[50],
                 sensor_branch_layers=7*[20],
                 receiver_branch_layers=7*[20],
                 root_layers=[100, 80, 60, 40, 20],
                 actf='tanh'):
        super(CA_DeepONet, self).__init__()
        
        # 定义输入层（核心修正：显式绑定到父类）
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.r_branch_inputs  = Input(shape=(receiver_size,), name='rb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')
        def global_context_block(input_feature, ratio=4):
            channel = input_feature.shape[-1]  # 输入特征维度
    
            # 全局平均池化（沿特征轴）
            x = tf.reduce_mean(input_feature, axis=1, keepdims=True)  # 形状: (batch, 1)
    
            # 全连接层压缩与恢复
            x = Dense(channel // ratio, activation='relu')(x)  # 压缩
            x = Dense(channel, activation='sigmoid')(x)        # 恢复
    
            # 注意力权重与特征相乘
            return Multiply()([input_feature, x])
        # sensor分支网络
        self.sensor_branch_scales = []
        x = self.branch_input
        for units in sensor_branch_layers:
            x = Dense(units, activation=actf)(x)
            x = global_context_block(x)
            self.sensor_branch_scales.append(x)
        
        self.receiver_branch_scales = []
        x = self.r_branch_inputs
        for units in receiver_branch_layers:
            x = Dense(units, activation=actf)(x)
            x = global_context_block(x)
            self.receiver_branch_scales.append(x)
        
        # 主干网络
        trunk_input = concatenate([self.trunk_input_x, self.trunk_input_y])
        self.trunk_scales = []
        x = trunk_input
        for units in trunk_layers:
            x = Dense(units, activation=actf)(x)
            self.trunk_scales.append(x)
        
        # --- 跨尺度特征融合 ---
        aligned_features = []
        for i in range(min(len(self.trunk_scales),len(self.sensor_branch_scales))):
            # 融合当前层 branch[i] 和 trunk[i]
            fused_current = concatenate([self.sensor_branch_scales[i], self.trunk_scales[i]])
            aligned_current = Dense(embedding_size, activation=actf)(fused_current)
            aligned_features.append(aligned_current)
        for i in range(min(len(self.trunk_scales),len(self.receiver_branch_scales))):
            # 融合当前层 branch[i] 和 trunk[i]
            fused_current = concatenate([self.receiver_branch_scales[i], self.trunk_scales[i]])
            aligned_current = Dense(embedding_size, activation=actf)(fused_current)
            aligned_features.append(aligned_current)
        
        # 全局注意力（保留）
        fused_feature = concatenate(aligned_features)
        fused_feature = global_context_block(fused_feature)
        
        # --- Root网络（无需修改）---
        final_feature = concatenate([fused_feature, self.sensor_branch_scales[-1]*self.trunk_scales[-1],self.receiver_branch_scales[-1]*self.trunk_scales[-1]])
        final_feature_aligned = Dense(root_layers[0])(final_feature)
        
        self.root = final_feature_aligned
        for idx, units in enumerate(root_layers):
            self.root = Dense(units, activation=actf)(self.root)
            if idx == 2 and self.root.shape[-1] == final_feature_aligned.shape[-1]:
                self.root = Add()([self.root, final_feature_aligned])
        # 输出层（直接绑定到父类）
        self.output_layer  = Dense(1, activation='linear', name='u')(self.root) 
        
        # 显式定义模型输入输出（关键修正！）
        super().__init__(
            inputs=[self.trunk_input_x, self.trunk_input_y,  self.branch_input , self.r_branch_inputs], 
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
        (x_trunk, y_trunk, t_branch , r_branch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch , r_branch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(v)) 
            
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
    def __init__(self, receiver_size=20,
                 sensor_size=20, 
                 embedding_size=100,
                 trunk_layers=[50],
                 sensor_branch_layers=7*[20],
                 receiver_branch_layers=7*[20],
                 root_layers=[100, 80, 60, 40, 20],
                 actf='tanh'):
        super(Root_DeepONet, self).__init__()
        
        # 定义输入层（核心修正：显式绑定到父类）
        self.branch_input  = Input(shape=(sensor_size,), name='vb')
        self.r_branch_inputs  = Input(shape=(receiver_size,), name='rb')
        self.trunk_input_x  = Input(shape=(1,), name='xt')
        self.trunk_input_y  = Input(shape=(1,), name='yt')
        # 构建网络分支 
        self.sensor_branch_net  = self._build_network(self.branch_input, sensor_branch_layers, actf, embedding_size)
        self.receiver_branch_net  = self._build_network(self.r_branch_inputs, receiver_branch_layers, actf, embedding_size)
        self.trunk_net  = self._build_network(
            concatenate([self.trunk_input_x, self.trunk_input_y]), 
            trunk_layers, actf, embedding_size 
        )
        
        # 特征交互层 
        self.multiply1  = Multiply()([self.trunk_net, self.sensor_branch_net]) 
        self.add1  = Add()([self.trunk_net, self.sensor_branch_net]) 
        self.subtract1  = Subtract()([self.trunk_net, self.sensor_branch_net]) 
        self.multiply2  = Multiply()([self.trunk_net, self.receiver_branch_net]) 
        self.add2  = Add()([self.trunk_net, self.receiver_branch_net]) 
        self.subtract2  = Subtract()([self.trunk_net, self.receiver_branch_net]) 
        # Root网络构建 
        self.root  = concatenate([self.multiply1, self.add1,  self.subtract1 ,self.multiply2 , self.add2 , self.subtract2]) 
        for units in root_layers:
            self.root  = Dense(units, activation=actf)(self.root) 
        
        # 输出层（直接绑定到父类）
        self.output_layer  = Dense(1, activation='linear', name='u')(self.root) 
        
        # 显式定义模型输入输出（关键修正！）
        super().__init__(
            inputs=[self.trunk_input_x, self.trunk_input_y,  self.branch_input , self.r_branch_inputs], 
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
        (x_trunk, y_trunk, t_branch , r_branch), y_true = data 
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播（直接调用call方法）
            y_pred = self([x_trunk, y_trunk, t_branch , r_branch], training=True)
            
            # 数据拟合损失 
            data_loss = self.compiled_loss(y_true,  y_pred)
            
            # 计算物理约束梯度（Eikonal方程）
            with tape.stop_recording(): 
                du_dx = tape.gradient(y_pred,  x_trunk)
                du_dy = tape.gradient(y_pred,  y_trunk)
            
            # Eikonal残差（假设VELMODEL为全局变量）
            eikonal_loss = tf.reduce_mean(du_dx**2  + du_dy**2 - 1/tf.square(v)) 
            
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
    def __init__(self,num_sample = 100, 
                    sample_size = 10, 
                    batch_size = dict(data=1000, domain=1000, bc=100),
                    recevier_size = 10,
                    sensor_size = 10,                       
                    shuffle = True,
                    seed=1234):
        # generate data
        self.num_sample = num_sample
        self._sample_size = sample_size
        self._batch_size = batch_size
        self._sensor_size = sensor_size
        sensor_size_1d = int(np.sqrt(self._sensor_size) + 0.001)
        grid_1d_x = np.linspace(XMIN, XMAX, sensor_size_1d + 2)[1:-1]
        grid_1d_y = np.linspace(YMIN, YMAX, sensor_size_1d + 2)[1:-1]
        self._sensor = [gi.flatten() for gi in np.meshgrid(grid_1d_x, grid_1d_y)]
        
        self._recevier_size = recevier_size
        self._receiver = [np.linspace(XMIN, XMAX, recevier_size+2)[1:-1], 
                          np.full(recevier_size, YMAX)]
        self._shuffle = shuffle
        self._epoch = 1
        self._time = time.time()
        self._velocity_ids = np.arange(0, num_sample, dtype=int)
        # self._velocity_ids = np.random.choice(len(vel_models), sample_size, replace=False)
        self._vel_ratio = 1.0
        self._set_data()
        
    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return int((self.num_sample - 1) / self._sample_size) + 1
    
    def __getitem__(self, index):
        start = index * self._sample_size
        end = min(start + self._sample_size, self.num_sample)
        idx = np.hstack(self._sample_ids[start:end])
        inputs = [xs[idx] for xs in self.inputs]
        outputs = [ys[idx] for ys in self.targets]
        sample_weights = [ws[idx] for ws in self.sample_weights]
        return inputs, outputs, sample_weights
    def get_sample(self, index):
        idx = self._sample_ids[index]
        inputs = [xs[idx] for xs in self.inputs]
        outputs = [ys[idx] for ys in self.targets]
        sample_weights = [ws[idx] for ws in self.sample_weights]
        return inputs, outputs, sample_weights
    
    def on_epoch_end(self):
        if self._epoch % 100 == 0:
            # self._vel_ratio = min(1, self._vel_ratio + 0.01)
            print(f"{self._epoch} - {self._time - time.time()}s - data updated with vel-ratio = {self._vel_ratio} ")
            # self._set_data()
        if self._shuffle:
            np.random.shuffle(self._sample_ids)
        self._epoch += 1

    def _set_data(self):
        num_sample, batch_size = self.num_sample, self._batch_size
        inputs = OrderedDict()
        targets = OrderedDict()
        sample_weights = OrderedDict()
        sample_ids = []
        total_sample_size = 0
        for i, idx in enumerate(self._velocity_ids):
            x_source = np.random.uniform(XMIN+0.15*DELTAX, XMAX-0.15*DELTAX)
            y_source = np.random.uniform(YMIN+0.15*DELTAY, YMAX-0.15*DELTAY)
            sample_velocity = vel_models[idx, 0]
            # scheduled_velocity = (sample_velocity - sample_velocity.mean()) * self._vel_ratio + sample_velocity.mean()
            inputs_i, targets_i, sample_weights_i = self.__generate_batch_data(
                x_source, y_source, sample_velocity, batch_size)
            num_sample_i = inputs_i[0].shape[0]
            for i_xs, xs in enumerate(inputs_i):
                if i_xs not in inputs:
                    inputs[i_xs] = []
                inputs[i_xs].append(xs)
            for i_ys, ys in enumerate(targets_i):
                if i_ys not in targets:
                    targets[i_ys] = []
                targets[i_ys].append(ys)
            for i_ws, ws in enumerate(sample_weights_i):
                if i_ws not in sample_weights:
                    sample_weights[i_ws] = []
                sample_weights[i_ws].append(ws)
            sample_ids.append(np.arange(num_sample_i).astype(int) + total_sample_size)
            total_sample_size = total_sample_size + num_sample_i
        # concat data
        self.inputs = [np.vstack(inputs[k]) for k in inputs]
        self.targets = [np.vstack(targets[k]) for k in targets]
        self.sample_weights = [np.hstack(sample_weights[k]) for k in sample_weights]
        self._sample_ids = sample_ids

    def __generate_batch_data(self, x_source, y_source, velocity, batch_size):
        counter = 0
        
        # eikonal (travel-time) solution
        eik2d = EikonalSolver(vel=velocity, xd=[XMIN, XMAX], yd=[YMIN, YMAX], source=[x_source, y_source])
        x_grid, y_grid = eik2d.xg
        
        # sample data
        if batch_size['data'] == 'all':
            ids_data = np.arange(0, x_grid.size, dtype=int)
        else:
            ids_data = np.random.choice(x_grid.size, batch_size['data'], replace=False)
        x_data, y_data = x_grid.flatten()[ids_data], y_grid.flatten()[ids_data]
        target_data = eik2d.grid.flatten()[ids_data]
        ids_data = np.arange(ids_data.size) + counter
        counter += ids_data.size

        # sample bc
        DELTAX = XMAX - XMIN
        DELTAY = YMAX - YMIN
        x_bc, y_bc = lhs_uniform(2, batch_size['bc'], 
                                 [[x_source - 1e-6, x_source + 1e-6], 
                                  [y_source - 1e-6, y_source + 1e-6]])
        target_bc = np.zeros_like(x_bc)
        ids_bc = np.arange(batch_size['bc']) + counter
        counter += ids_bc.size

        # contact data
        x_data = np.hstack([x_data, x_bc]).flatten()
        y_data = np.hstack([y_data, y_bc]).flatten()
        target_data = np.hstack([target_data, target_bc])
        ids_data = np.hstack([ids_data, ids_bc]).flatten()
        
        size_sample = counter
        
        # trunk inputs
        x_trunk = np.hstack([x_data]).reshape(-1,1)
        y_trunk = np.hstack([y_data]).reshape(-1,1)
        
        # - BRANCH -
        # velocity-branch inputs
        x_sensor, y_sensor = self._sensor
        v_sensor = interpolate_velocity_model(velocity, x_sensor, y_sensor)
        v_branch = np.tile(v_sensor, (size_sample, 1))
        
        # recevier-branch inputs
        x_receiver, y_receiver = self._receiver
        t_receiver = eik2d(x_receiver, y_receiver)
        t_branch = np.tile(t_receiver, (size_sample, 1))

        # inputs
        inputs = [x_trunk, y_trunk, v_branch, t_branch]
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
        
        return inputs, targets, sample_weights
    
    
    def generate_test_data(self, Xs, Ys, Vs, Nx=100, Ny=100, sensor_size=10, noise=0.0):
        counter = 0
        
        # eikonal solution
        eik2d = EikonalSolver(vel=Vs, xd=[XMIN, XMAX], yd=[YMIN, YMAX], source=[Xs, Ys])
        x_grid, y_grid = eik2d.xg
        t_grid = eik2d.grid
        
        # sample domain
        x_trunk, y_trunk = x_grid.reshape(-1,1), y_grid.reshape(-1,1)
        v_trunk = Vs.reshape(-1,1)
        target = t_grid.reshape(-1,1)
        size_sample = Nx*Ny
        
        # velocity-branch inputs
        x_sensor, y_sensor = self._sensor
        v_sensor = interpolate_velocity_model(Vs, x_sensor, y_sensor)
        v_branch = np.tile(v_sensor, (size_sample, 1))
        
        # recevier-branch inputs
        x_receiver, y_receiver = self._receiver
        t_receiver = eik2d(x_receiver, y_receiver)
        
        if noise > 0:
            t_receiver += noise*np.std(t_receiver)*np.random.normal(0, 1, t_receiver.shape)
            t_receiver = np.maximum(0, t_receiver)
            
        t_branch = np.tile(t_receiver, (size_sample, 1))

        # inputs
        inputs = [x_trunk, y_trunk, v_branch, t_branch]
        
        return inputs, v_trunk, target
    
    

# %%
dg = DataGenerator(
    num_sample=400,
    sample_size=25,
    batch_size={'data': 'all', 'bc':200},
    sensor_size=SENSOR_SIZE,
    recevier_size=RECEIVER_SIZE
)

# %%
from itertools import cycle

n_plot = 4
fig, ax = plt.subplots(1, n_plot, figsize=(15, 3))
for i in range(n_plot):
    cycol = cycle('brycmg')
    inputs, targets, weights = dg[np.random.choice(len(dg))]
    for wi in weights:
        x_trunk, y_trunk, v_branch,t_branch = inputs
        idx = wi > 0
        ax[i].scatter(x_trunk[idx], y_trunk[idx], color=next(cycol), alpha=0.2)

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
    receiver_size=RECEIVER_SIZE,
    embedding_size=50,
    trunk_layers=5*[50],
    sensor_branch_layers=[100,80,50,50,50],
    receiver_branch_layers=5*[50],
    root_layers=[100, 80, 60, 40, 20],
    actf='tanh'
)
ROOT_DON = Root_DeepONet(
    sensor_size=SENSOR_SIZE,
    receiver_size=RECEIVER_SIZE,
    embedding_size=50,
    trunk_layers=5*[50],
    sensor_branch_layers=[100,80,50,50,50],
    receiver_branch_layers=5*[50],
    root_layers=[100, 80, 60, 40, 20],
    actf='tanh'
)
DON = DeepONet(
    sensor_size=SENSOR_SIZE,
    receiver_size=RECEIVER_SIZE,
    embedding_size=50,
    trunk_layers=5*[50],
    sensor_branch_layers=[100,80,50,50,50],
    receiver_branch_layers=5*[50],
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
checkpoint_ROOT = ModelCheckpoint(
    filepath=os.path.join(PATH_NAME, "weights_ROOT-epoch{epoch:06d}.hdf5"),
    save_weights_only=True,
    period=SAVE_PERIOD,
    verbose=1
)
checkpoint_DON = ModelCheckpoint(
    filepath=os.path.join(PATH_NAME, "weights_DON-epoch{epoch:06d}.hdf5"),
    save_weights_only=True,
    period=SAVE_PERIOD,
    verbose=1
)
lr_callback = LearningRateScheduler(lr_scheduler)
loss_logger = LossLogger(
    log_dir=PATH_NAME, 
    file_name="physics_informed_lossca.csv", 
    save_freq=1  # 每100个epoch保存一次 
)
loss_logger2 = LossLogger(
    log_dir=PATH_NAME, 
    file_name="physics_informed_lossroot.csv", 
    save_freq=1  # 每100个epoch保存一次 
)
loss_logger3 = LossLogger(
    log_dir=PATH_NAME, 
    file_name="physics_informed_lossdon.csv", 
    save_freq=1  # 每100个epoch保存一次 
)
conditioned_lr = ConditionedLRScheduler(check_interval=100, patience=3, factor=0.5, min_lr=1e-6)
# 训练模型时添加回调 
""" CA_DON.fit( 
    dg,
    epochs=5000,  # 示例：总训练轮次设为1000 
    callbacks=[loss_logger,lr_callback,conditioned_lr,checkpoint_CA],
    verbose=1 
)
ROOT_DON.fit( 
    dg,
    epochs=5000,  # 示例：总训练轮次设为1000 
    callbacks=[loss_logger2,lr_callback,conditioned_lr,checkpoint_ROOT],
    verbose=1 
)
DON.fit( 
    dg,
    epochs=5000,  # 示例：总训练轮次设为1000 
    callbacks=[loss_logger3,lr_callback,conditioned_lr,checkpoint_DON],
    verbose=1 
) """
#DON.fit(dg, epochs=5000, verbose=0)
# [x.shape for x in dg[0][0]]

# %%
#DON.plot_loss()
# %%
#np.random.seed(1234)
weight_path_CA = os.path.join(PATH_NAME, 'weights_CA-epoch005000.hdf5')
CA_DON.load_weights(weight_path_CA)
weight_path_ROOT = os.path.join(PATH_NAME, 'weights_ROOT-epoch005000.hdf5')
ROOT_DON.load_weights(weight_path_ROOT)
weight_path_DON = os.path.join(PATH_NAME, 'weights_DON-epoch005000.hdf5')
DON.load_weights(weight_path_DON)
#Nx, Ny = 70, 70
""" perf_logger = PerformanceLogger(file_name='test_metrics_CA.csv')
perf_logger2 = PerformanceLogger(file_name='test_metrics_ROOT.csv')
perf_logger3 = PerformanceLogger(file_name='test_metrics_DON.csv')
for i in range(10):
    velocity = vel_models[300 + i, 0]
    x_source = np.random.uniform(XMIN+0.1*DELTAX, XMAX-0.1*DELTAX)
    y_source = np.random.uniform(YMIN+0.1*DELTAY, YMAX-0.1*DELTAY)
    
    test_data, vel_data, target_data = dg.generate_test_data(x_source, y_source, velocity, Nx=Nx, Ny=Ny, sensor_size=SENSOR_SIZE)

    x_test = test_data[0].reshape(Nx, Ny)
    y_test = test_data[1].reshape(Nx, Ny)
    v_test = np.copy(velocity)
    exact = target_data.reshape(Nx, Ny)
        
    pred = CA_DON.predict(test_data).reshape(Nx, Ny)
    pred2 = ROOT_DON.predict(test_data).reshape(Nx, Ny)
    pred3 = DON.predict(test_data).reshape(Nx, Ny)
    PerformanceLogger.record_metrics(
        perf_logger.log_path,
        sample_id=i,
        x_source=x_source,
        y_source=y_source,
        y_true=exact.flatten(),
        y_pred=pred.flatten()
    )
    PerformanceLogger.record_metrics(
        perf_logger2.log_path,
        sample_id=i,
        x_source=x_source,
        y_source=y_source,
        y_true=exact.flatten(),
        y_pred=pred2.flatten()
    )
    PerformanceLogger.record_metrics(
        perf_logger3.log_path,
        sample_id=i,
        x_source=x_source,
        y_source=y_source,
        y_true=exact.flatten(),
        y_pred=pred3.flatten()
    )
    fig, ax = plt.subplots(1, 5, figsize=(16, 3))
    
    # ax0 = ax[0].pcolor(x_test, y_test, v_test, cmap='seismic')
    # ax[0].set_title('velocity field')
    # ax[0].axis('off')
    # plt.colorbar(ax0, ax=ax[0])
    ax = [None, ax[0], ax[1], ax[2], ax[3],ax[4]]
    ax1 = ax[1].contour(x_test, y_test, pred, cmap='seismic', vmin=-1, vmax=1)
    ax[1].scatter(x_source, y_source, marker='*', color='r')
    ax[1].set_title('pred')
    ax[1].axis('off')
    plt.colorbar(ax1, ax=ax[1])
    
    ax2 = ax[2].contour(x_test, y_test, exact, cmap='seismic', vmin=-1, vmax=1)
    ax[2].scatter(x_source, y_source, marker='*', color='r')
    ax[2].set_title('exact')
    ax[2].axis('off')
    plt.colorbar(ax2, ax=ax[2])
    
    ax3 = ax[3].pcolor(x_test, y_test, abs(pred - exact), cmap='seismic', vmin=-1, vmax=1)
    ax[3].set_title('|predCA - true|')
    ax[3].axis('off')
    plt.colorbar(ax3, ax=ax[3])

    ax4 = ax[4].pcolor(x_test, y_test, abs(pred2 - exact), cmap='seismic', vmin=-1, vmax=1)
    ax[4].set_title('|predROOT - true|')
    ax[4].axis('off')
    plt.colorbar(ax4, ax=ax[4])
    ax5 = ax[5].pcolor(x_test, y_test, abs(pred3 - exact), cmap='seismic', vmin=-1, vmax=1)
    ax[5].set_title('|predDON - true|')
    ax[5].axis('off')
    plt.colorbar(ax5, ax=ax[5])
    #plt.show()
    plt.savefig(os.path.join(PATH_NAME, "lbfgs_loss.png")) """
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
Nx, Ny = 70, 70

rmse_error = []
rmse_errorDON = []
rmse_errorRoot = []
mae_error = []      # 新增：存储 MS-DeepONet 的 MAE
mae_errorDON = []   # 新增：存储 DON 的 MAE
mae_errorRoot = []
r2_error = []       # 新增：存储 MS-DeepONet 的 R²
r2_errorDON = []    # 新增：存储 DON 的 R²
r2_errorRoot =[]
source_locs = []
source_error = []

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
titles = ['Velocity Field (km/s)', 'FNN: T(s)', 'MF-DeepONet: T(s) Our', 'DON: T(s)', '|MF-DON Error| Our', '|DON Error|']
fig, ax = plt.subplots(5, 6, figsize=(32, 15))
for i in range(NUM_TRIAL):  
    velocity = vel_models[i, 0]
    x_source = np.random.uniform(XMIN+0.1*DELTAX, XMAX-0.1*DELTAX)
    y_source = np.random.uniform(YMIN+0.1*DELTAY, YMAX-0.1*DELTAY)

    
    test_data, vel_data, target_data = dg.generate_test_data(
        x_source, y_source, velocity, Nx=Nx, Ny=Ny, sensor_size=SENSOR_SIZE)

    x_test = test_data[0].reshape(Nx, Ny)
    y_test = test_data[1].reshape(Nx, Ny)
    v_test = vel_data.reshape(Nx, Ny)
    
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
    source_locs.append([x_source, y_source])
    
    source_error.append(
        [x_source-y_source_pred, y_source-x_source_pred]
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
        cust_pcolor(ax[num,0], x_test, y_test, v_test.T, title='Velocity Field (km/s)' if num==0 else None, cmap='rainbow', xlabel=True if num==4 else False,ylabel=True,yticks = False)
        cust_pcolor(ax[num,1], x_test, y_test, exact, x_source, y_source, 'FNN: T(s)' if num==0 else None, 'rainbow',xlabel=True if num==4 else False,ylabel=False,yticks = False)
        cust_pcolor(ax[num,2], x_test, y_test, pred, x_source_pred, y_source_pred, 'MF-DeepONet: T(s) OURS' if num==0 else None, 'rainbow',
                    vmin=exact.min(), vmax=exact.max(),xlabel=True if num==4 else False, ylabel=False,yticks = False)
        cust_pcolor(ax[num,3], x_test, y_test, predDON, x_source_pred_DON, y_source_pred_DON, 
                    'DON: T(s)' if num==0 else None, 'rainbow', vmin=exact.min(), vmax=exact.max(),xlabel=True if num==4 else False, ylabel=False,yticks = False)
        cust_pcolor(ax[num,4], x_test, y_test, abs(pred - exact), title='|MF-DON Error| OURS' if num==0 else None, cmap='rainbow',vmax=error_max,xlabel=True if num==4 else False,ylabel = False,yticks = False)
        cust_pcolor(ax[num,5], x_test, y_test, abs(predDON - exact), title='|DON Error|' if num==0 else None, cmap='rainbow',vmax=error_max,xlabel=True if num==4 else False, ylabel = False,yticks = False)
        num=num+1
font_tnr = FontProperties(family='Times New Roman', size=12)
df_metrics = pd.DataFrame(metrics)
csv_path = os.path.join(PATH_NAME, "model_metrics.csv")
df_metrics.to_csv(csv_path, index=False)
plt.subplots_adjust(0.05, 0.08, 0.98, 0.94, 0.1, 0.3)
plt.savefig(os.path.join(PATH_NAME, f'preds.png'))
rmse_error = np.array(rmse_error)
rmse_errorDON = np.array(rmse_errorDON)
weights = np.ones(len(rmse_error)) / len(rmse_error) * 100
weightsDON = np.ones(len(rmse_errorDON)) / len(rmse_errorDON) * 100
bins = np.linspace(0, 0.1, 20)
plt.figure(figsize=(4,3))
plt.hist(rmse_error, bins=bins,weights=weights, 
         alpha=0.5, color='blue', label='MF-DON')
# 绘制第二个直方图
plt.hist(rmse_errorDON,bins=bins,weights=weightsDON, 
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

