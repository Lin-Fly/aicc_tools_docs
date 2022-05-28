
# AICC Tools 使用教程

## AICC Tools 介绍

​		AICC Tools 是一款面向MindSpore开发者和华为人工智能计算中心，致力于**帮助开发者快速将代码适配至计算中心，易化和规范化云上文件交互过程，简化云上日志显示，以此加速模型开发进度和降低云上集群调试难度**。

​		其包含四大功能：

​		**1. ”零代码“迁移适配计算中心，裸机和AICC环境通用；**

​		**2.  云上日志分卡/分类/打屏和保存；**

​		**3.  云上文件异步交互传输；**

​		**4.  云上分布式训练“一键”启动；**

​		aicc tools 在使用层面上完全兼容了本地运行环境和AICC运行环境，代码线下线上可做到一体化，不仅能够帮助开发者便捷、快速的将本地（线下）代码迁移至计算中心，还可以一行代码解决云上分布式训练问题。利用云上文件交互传输功能，可解决云上大规模模型或大规模数据实时交互传输问题。简洁的日志显示和分类系统也极大地便利了开发者利用云上集群调试大模型。

## 本地安装

```shell
pip install aicc_tools
```

## ModelArts安装

```tex
以训练作业服务为例：
1. 在obs桶内准备需要训练的工程文件，工程目录如下：
	obs://your_bucket/work/your_project
2. 在obs://your_bucket/work/your_project 目录下放置 aicc_tools 离线wheel包
3. 在your_project目录下编写名为 ma-pre-start.sh 脚本， 内容如下：
	ma-pre-start.sh：
    echo ***********ma-pre-start*********
    cd /home/work/user-job-dir/your_project
    pip install aicc_tools-0.1.6-py3-none-any.whl --ignore-installed
    echo ***********ma-pre-end*********
```

ma-pre-start.sh脚本是ModelArts在拉起容器训练后，首先会自动执行的脚本，在这里用户可以指定运行的环境或者安装第三方wheel包

# AICC Tools 全流程使用

## 快速入门体验

[MindSpore模型算法快速适配AICC](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#demo1-mindspore)

## 一、AICC 训练进程监控

aicc tools 中提供了 aicc_monitor装饰器接口，主要用于帮助开发者在AICC平台上监控训练算法的进程是否发生异常，在训练正常结束或者异常中断后，补获日志信息，同时将用户产生的最后一份文件回传保存至obs桶中。主要保存的内容包括：日志（plog、用户打印日志、错误日志、mindspore日志等）、模型权重文件、summary文件、其他用户保存的文件等。如果用户想要保存mindspore相关的图信息（graph、ir、dot、dump文件等），请参考MindSpore环境设置进行操作：[MindSpore运行环境设置](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#mindspore)

### aicc_monitor 使用样例：

```python
import aicc_tools as ac

@ac.aicc_monitor
def main():
    ....

if __name__ == "__main__":
    main()
```

## 二、AICC 分布式集群训练启动

aicc tools为MindSpore 开发者封装了运行环境配置的高阶API接口：context_init(**kwargs)，可以帮助开发者快速使用MindSpore来适配AICC平台进行集群多机多卡的分布式训练启动，且支持线下物理机使用，无需修改代码。

### context_init 参数说明：

```tex
Args:
    seed: 
    	设置全局随机种子数, 默认0
    use_parallel: 
    	是否使用分布式训练，默认是True
    context_config: mindspore基本环境初始化配置。支持mindspore.context.set_context(**kwargs)内所有支持的参数配置。
    	默认在Ascend上的静态图模式执行，context_config = {"mode": "GRAPH_MODE", "device_target": "Ascend", "device_id": 0}
    
    parallel_config: mindspore并行环境初始化配置。支持mindspore.context.set_auto_parallel_context(**kwargs)内所有支持的参数配置
    	默认数据并行模式，parallel_config = {"parallel_mode": "DATA_PARALLEL", "gradients_mean": True}
Return:
	rank_id: 当前程序执行的device编号，如：0,1,....
	device_num: 当前程序使用的总的device数量， 如device_num = 8（单机8卡）
```

### MindSpore 运行环境设置

```python
import aicc_tools as ac
# Example1: 默认数据并行方式启动
rank_id, device_num = ac.context_init(seed=0)

# Example2: 自定义并行方式启动: 数据并行模式
## 其中mode参数: 0 代表静态图模式GRAPH_MODE，1 代表动态图模式PYNATIVE_MODE
context_config = {"mode": 0, "device_target": "Ascend", "device_id": 0}
## 其中parallel_mode: 0 代表数据并行模式DATA_PARALLEL，1 代表半自动并行模式SEMI_AUTO_PARALLEL，2 代表自动并行模式AUTO_PARALLEL
parallel_config = {"parallel_mode": 0, "gradients_mean": True}
rank_id, device_num = ac.context_init(seed=0, parallel=True, context_config=context_config, parallel_config=parallel_config)
```

#### 保存图信息或Dump信息

aicc tools 可通过封装的环境初始化函数 context_init(**kwargs)函数开启图信息或Dump信息保存，AICC环境无需指定保存路径，将自动回传至CFTS系统指定的obs_path路径。（注此处必须配套使用[CFTS系统](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#id6)和[aicc_monitor监控](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#aicc)）

```python
import aicc_tools as ac
# 自定义并行方式启动: 保存图，用于调试定位错误信息
context_config = {
    "mode": 0,
    "device_target": "Ascend",
    "device_id": 0,
    "save_graphs": True,
    "enable_dump": True
}  # AICC平台适配只需开启save_graphs或enable_dump,无需指定路径
parallel_config = {"parallel_mode": 0, "gradients_mean": True}
rank_id, device_num = ac.context_init(
    seed=0, parallel=True, context_config=context_config, parallel_config=parallel_config)
```

## 三、AICC 文件异步交互系统

aicc tools 为用户提供了规范、便捷和易用的AICC平台文件上传和载入的API接口：CFTS，帮助用户可以将代码快速适配至AICC平台。其最大特点是能够帮助用户在代码层面将文件传输的过程隐去，让用户可以无感知的进行文件交互操作，实现 OBS存储==物理机磁盘的转换过程，且传输过程采用异步多进程方式，不仅可以不增加额外训练成本，还能最大程度上降低用户适配和使用AICC平台的难度。

### CFTS API 参数说明：

```tex
Args:
	obs_path: 用于存储云上训练过程中产生的所有数据的obs桶路径， 通常是以obs或s3开头的链接， 默认值：None
	rank_id: 指定回传rank_id对应的数据, 默认按照一个节点只回传一张卡的数据, 默认值：None
	upload_frequence：回传数据至obs桶的频次，默认是训练一个step回传一次数据，默认值：1
	keep_last：保持始终是最新的文件，删掉旧文件，默认值：True
return：
	CFTS的class实例
```

#### 初始化使用样例：

```python
import aicc_tools as ac
# Example1: 按照每训练1step回传0卡的相关数据至obs路径中,obs中数据与集群节点保持一致，清除旧文件
cfts_1 = ac.CFTS(obs_path="obs存储路径", rank_id=0, upload_frequence=1, keep_last=True)

# Example2: 按照每训练1step回传每个节点的第0卡相关数据至obs路径中,obs中数据与集群节点保持一致，清除旧文件
cfts_2 = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=True)

# Example3: 按照每训练1step回传每个节点的第0卡相关数据至obs路径中,obs中数据累积不清除
cfts_3 = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
```


### CFTS 对外提供接口：

* **[get_dataset](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#get-dataset)**：载入数据集，并返回存放路径
* **[get_checkpoint](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#get-checkpoiint)**：载入权重文件，并返回存放路径
* **[obs_monitor](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#obs-monitor-obs)**：
  * mindspore中callback函数类，用于model.train(callback=[..，checkpoint_monitor()，obs_mointor()])，保证用户文件被保存至用户指定的obs路径中
* **[checkpoint_monitor](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#checkpoint-monitor)**：
  * mindspore中callback函数类，用于model.train(callback=[.., checkpoint_monitor()])，保存模型权重文件
* **[loss_monitor](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#loss-monitor)**：
  * mindspore中callback函数类，用于model.train(callback=[.., loss_monitor()])，打印训练过程中的loss
* **[summary_monitor](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#summary-monitor)**：
  * mindspore中callback函数类，用于model.train(callback=[.., summary_monitor()])，保存训练过程中的summary文件
* **[get_custom_path](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#get-custom-path)**：AICC平台默认文件存储根路径，所有用户要保存的文件应存放在该目录下，适用于用户想要**自定义保存**一些文件时使用
* **[send2obs](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#send2obs-obs)**：用户自定义回传接口，该接口支持用户自定义训练过程中的文件回传存储至obs中


#### get_dataset 数据集载入

get_dataset 帮助用户可以在代码层面上像使用物理机磁盘一样使用obs存储服务来加载数据集，仅需要传入数据集所在的obs路径（AICC平台时）或者本地路径，系统会自动帮助用户将数据集从obs中拉取到集群容器上，并返回实际存放的路径给用户，整个过程用户不在感知，和线下物理机使用无差异。

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
# Example1：线下物理机使用
data_path = cfts.get_dataset(dataset_path="磁盘数据集路径")
# Example2：AICC平台使用
data_path = cfts.get_dataset(dataset_path="obs数据集路径")
```


#### get_checkpoiint 权重文件载入

get_checkpoiint 帮助用户可以在代码层面上像使用物理机磁盘一样使用obs存储服务来加载模型权重，仅需要传入模型权重所在的obs路径，系统会自动帮助用户将模型权重从obs中拉取到集群容器上，并返回实际存放的路径给用户，整个过程用户不在感知，和线下物理机使用无差异。

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
# Example1：线下物理机使用
ckpt_path = cfts.get_checkpoint(dataset_path="磁盘模型路径")
# Example2：AICC平台使用
ckpt_path = cfts.get_checkpoint(dataset_path="obs模型路径")
```


#### obs_monitor 文件回传保存obs

obs_monitor 函数为用户在AICC平台专门适配了规范的文件实时传输功能，可以帮助用户无感知的将在AICC平台上训练保存的文件保存到obs中，其采用异步传输特性，传输的时间损耗可以由训练时间所掩盖，不会给用户增加额外的训练成本。**值得注意的是，obs_monitor 函数目前只适用于MindSpore框架中，且需要采用model.train() or model.eval() 的方式传入callbacks=[..., obs_monitor()] 参数方式使用，且需要放在callbacks列表的最后！**

请参考以下接口使用：[保存权重文件](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#checkpoint-monitor)  [收集训练信息](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#summary-monitor)  [保存自定义文件](https://aicc-tools-docs.obs.cn-southwest-228.cdzs.cn/instruction/aicc_tools_docs/build/html/aicc_tools_docs.html#get-custom-path)


#### checkpoint_monitor 保存权重

checkpoint_monitor 帮助用户封装了MindSpore的[ModelCheckpoint](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.train.html?highlight=callback#mindspore.train.callback.ModelCheckpoint), [CheckpointConfig](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.train.html?highlight=callback#mindspore.train.callback.CheckpointConfig)，支持其中所有超参，可以便捷的帮助用户在AICC上保存模型权重文件。

**checkpoint_monitor 参数接口说明（注：引用自mindspore源码）：**

```tex
Args:
	prefix (str): The prefix name of checkpoint files. Default: "CKP".
    directory (str): The path of the folder which will be saved in the checkpoint file.
    By default, the file is saved in the current directory. Default: None.
    save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
    save_checkpoint_seconds (int): Seconds to save checkpoint.
    Can't be used with save_checkpoint_steps at the same time. Default: 0.
    keep_checkpoint_max (int): Maximum number of checkpoint files can be saved. Default: 5.
    keep_checkpoint_per_n_minutes (int): Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes.
    Can't be used with keep_checkpoint_max at the same time. Default: 0.
    integrated_save (bool): Whether to merge and save the split Tensor in the automatic parallel scenario.
    Integrated save function is only supported in automatic parallel scene, not supported
    in manual parallel. Default: True.
    async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False.
    saved_network (Cell): Network to be saved in checkpoint file. If the saved_network has no relation
    with the network in training, the initial value of saved_network will be saved. Default: None.
    append_info (list): The information save to checkpoint file. Support "epoch_num", "step_num" and dict.
    The key of dict must be str, the value of dict must be one of int float and bool. Default: None.
    enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
    is not required. Default: None.
    enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
    mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.
    exception_save (bool): Whether to save the current checkpoint when an exception occurs. Default: False.
```

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
# directory 不设置时， 线下物理机环境将自动存放在 ./output/rank_*/checkpoint/， AICC平台将自动存放在默认路径下
ckpt_cb = cfts.checkpoint_monitor(prefix="aicc",
                                  directory="./ckpt", # 此处设置的路径只对线下物理机生效，AICC平台将自动存放在默认路径下
                                  save_checkpoint_steps=100,
                                  keep_checkpoint_max=1,
                                  integrated_save=False)
epoch = 10
dataset = None
model = None
# 需要注意的是，mindspore中所保存的文件均需要通过 cfts.obs_monitor() 接口执行进行回传，因此 callbacks列表中必须包含该callback函数！！！
model.train(epoch, dataset, callbacks=[ckpt_cb, cfts.obs_monitor()], dataset_sink_mode=True, sink_size=-1)
```


#### loss_monitor 训练损失打印

loss_monitor 函数用法与MindSpore的[LossMonitor](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.train.html?highlight=callback#mindspore.train.callback.LossMonitor)一致，不同之处在于为AICC平台适配了aicc tools中的日志系统，当在AICC执行分布式集群训练时，可避免将8张卡的数据同时打印至屏幕，进而简化打屏输出，方便用户查看。

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
loss_cb = cfts.loss_monitor(per_print_times=1)
epoch = 10
dataset = None
model = None
# 需要注意的是，mindspore中所保存的文件均需要通过 cfts.obs_monitor() 接口执行进行回传，因此 callbacks列表最后必须包含该callback函数！！！
model.train(epoch, dataset, callbacks=[loss_cb, cfts.obs_monitor()], dataset_sink_mode=True, sink_size=-1)
```


#### summary_monitor 训练信息收集

summary_monitor 函数用法与MindSpore的[SummaryCollector](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.train.html?highlight=callback#mindspore.train.callback.SummaryCollector)一致，不同之处在于专门为AICC平台进行了适配，所收集文件将自动回传保存至用户指定的obs目录中，用户无需感知。同时支持物理裸机环境，传入save_path参数可自定义保存位置。

**summary_monitor 参数说明：**

```tex
Args:
	summary_dir: 保存文件路径，物理机默认保存在当前目录下的output文件夹下, AICC环境默认存在回传路径下，默认值: None
	**kwargs: SummaryCollector支持的除summary_dir之外的其余参数
Return:
	class SummaryCollector
```

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)
# Example1: 指定summary保存路径，仅对线下物理机生效
summary_cb = cfts.summary_monitor(summary_dir='./summary', collect_freq=1) # 用户一般无需指定save_path
# Example2: 不指定summary保存路径，线下物理机默认保存在 ./output/rank_0/summary/, AICC环境自动存储到默认回传路径下
summary_cb = cfts.summary_monitor(collect_freq=1) # 用户一般无需指定save_path

epoch = 10
dataset = None
model = None
# 需要注意的是，mindspore中所保存的文件均需要通过 cfts.obs_monitor() 接口执行进行回传，因此 callbacks列表最后必须包含该callback函数！！！
model.train(epoch, dataset, callbacks=[summary_cb, cfts.obs_monitor()], dataset_sink_mode=True, sink_size=-1)
```


#### get_custom_path 自定义文件保存目录

get_custom_path 函数为用户在AICC平台上生成可用于自动回传保存的目录路径，方便用户将自定义保存的一些文件回传保存至用户指定的obs目录中，用户无需感知文件交互过程。

**get_custom_path 接口参数说明：**

```tex
Args:
	directory: 用户自定义的保存目录名称，当目录不存在时会自动创建，默认值：None
	file_name: 用户自定义保存的文件名称，默认值：None
Return:
	生成有效的文件保存路径
```

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)

# Example1: 用户自定义mindspore中callback函数，指定自定义保存文件夹
custom_cb_1 = CustomMonitor1(save_dir=cfts.get_custom_path(directory="custom1"))
# Example2: 用户自定义mindspore中callback函数，指定自定义保存文件夹和文件名称
custom_cb_2 = CustomMonitor2(save_dir=cfts.get_custom_path(directory="custom2", file_name="show.jpg"))

epoch = 10
dataset = None
model = None
# 需要注意的是，mindspore中所保存的文件均需要通过 cfts.obs_monitor() 接口执行进行回传，因此 callbacks列表最后必须包含该callback函数！！！
model.train(epoch, dataset, callbacks=[custom_cb_1, custom_cb_2, cfts.obs_monitor()], dataset_sink_mode=True, sink_size=-1)
```


#### send2obs 数据发送OBS

send2obs 为用户提供了规范、稳定的AICC平台文件回传接口， 方便用户调用直接回传已经被保存的各类文件，适用于自定义训练过程中手动回传文件。

**send2obs 接口参数说明：**

```tex
Args：
	src_url: AICC 集群容器中的本地路径，如/cache/ma-user-work/save_path, 默认: None
	obs_url: AICC OBS存储服务中的obs路径，以obs:// or s3:// 开头的路径， 默认: None
```

**使用样例：**

```python
cfts = ac.CFTS(obs_path="obs存储路径", upload_frequence=1, keep_last=False)

# Example1: 用户已经自定义保存过文件，且保存目录使用cfts.get_custom_path生成
cfts.send2obs() # 此时无需指定src_url和obs_url，文件默认回传到obs_path中
cfts.send2obs(obs_url="s3://new/custom") # 指定obs_url，文件则回传到obs_url中

# Example2: 用户已经定义过保存的文件，但保存目录未使用cfts.get_custom_path生成
cfts.send2obs(src_url="用户自定义的目录或文件") # 默认将该文件回传到obs_path中
cfts.send2obs(src_url="用户自定义的目录或文件", obs_url="s3://new/custom") # 将该文件回传到指定的obs_url中
```


## 四、AICC 日志系统

为了解决云上日志输出混乱的问题，AICC Tools 提供了线上线下行为自动切换、用户无感的日志控制系统，内部包括日志器生成器和重定向器。日志器生成器可根据用户的需要自定义日志器的行为，并返回一个日志器供用户输出日志。重定向器用于解决线上线下多卡训练模型时同时输出MindSpore日志的问题，当开启时，可以使指定节点的MindSpore日志输出到文件中，避免日志刷屏行为。

### get_logger 日志器生成器 

`get_logger` 用于生成日志器供用户输出日志使用。

```python
ac.get_logger(logger_name: str = 'aicc',
              to_std: bool = True,
              stdout_nodes: Union[List, Tuple, None] = None,
              stdout_device: Union[List, Tuple, None] = None,
              stdout_level: str = 'INFO',
              stdout_format: str = '',
              file_level: Union[List, Tuple] = ('INFO', 'ERROR'),
              file_save_dir: str = '',
              append_rank_dir: bool = True, 
              file_name: Union[List, Tuple] = ('aicc.log', 'error.log'),
              max_file_size: int = 50,
              max_num_of_files: int = 5) -> logging.Logger
```

- **logger_name**：日志器名称，当需要使用多个日志器的时候，需要保证多个日志器的名称不相同。

- **to_std**：日志是否需要输出到屏幕，默认为 `True`。

- **stdout_nodes**：当to_std为True时，该变量控制哪些计算节点可以输出日志到屏幕。默认为None，表示所以的计算节点都会输出日志到屏幕。当值设置为`[0, 1, 2, 3]`或者`(0, 1, 2, 3)`时，表示计算节点0、1、2和3中的程序可能会输出日志到屏幕。

- **stdout_device**：当to_std为True时，该变量控制一个计算节点中哪些卡对应程序可以输出日志到屏幕。默认为None，表示所以卡都会输出日志到屏幕。当值设置为`[0, 1, 2, 3]`或者`(0, 1, 2, 3)`时，表示卡0、1、2和3中的程序可能会输出日志到屏幕。

  当stdout_nodes值为[0, 1]，stdout_device值为[0, 1]时，表示计算节点0和1上的第0、1卡对应程序可以输出日志到屏幕。

- **stdout_level**：输出到屏幕的日志的等级。可设置为 `DEBUG`、`INFO`、`WARNING`、`ERROR` 和`CRITICAL`。当设置为 `INFO` 时，会输出 `INFO` 以上等级的日志到屏幕。

- **stdout_format**：输出到屏幕的日志的格式。默认为 `''`，表示选择默认的格式，即 `%(asctime)s - %(name)s - %(levelname)s - %(message)s`，具体设置可参考 [LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes)。

- **file_level**：输出到文件的日志的等级。默认为 `('INFO', 'ERROR')`，表示会将输出 INFO 和 ERROR 等级的日志到 `file_name` 中对应的文件。

- **file_save_dir**：日志文件保存的文件夹。默认为 `''`，表示选择默认的文件夹进行存储，线下运行时默认文件夹为 `./log/`，线上运行时默认文件夹为 `/cache/ma-user-work/`，且不随传值发生变化。

- **append_rank_dir**：是否在 `file_save_dir` 后添加与rank相关的文件夹。默认为 `True`，线下运行时路径变为 `./log/rank_0`，线上同理。

- **file_name**：保存日志的文件的名称列表。默认为 `('aicc.log', 'error.log')`，假设 `file_level` 为默认值，则会将 INFO 和 ERROR 等级的日志保存到 aicc.log 和 error.log 中。

- **max_file_size**：日志文件的最大大小。默认为 50，表示单个日志文件最大为 50 MB。

- **max_num_of_files**：最多保存的日志文件的数量。默认为 5，表示最多保存 5 个日志文件。

**使用样例：**

```python
import aicc_tools as ac

logger1 = ac.get_logger()
logger1.debug('debug test.')
logger1.info('info test.')
logger1.warning('warning test.')
logger1.error('error test.')
logger1.critical('critical test.')

logger2 = ac.get_logger('logger2', to_std=False, file_level=('INFO',), file_name=('info.log'))

# 如果您在其他文件夹中仍然需要使用logger2，则只需要：
logger2 = ac.get_logger('logger2')
```

### AiLogFastStreamRedirect2File 重定向器

`AiLogFastStreamRedirect2File` 本质为将流重定向到文件的重定向器，MindSpore 的日志默认输出到 `sys.stderr`，通过设置参数我们可以重定向输往 `sys.stderr` 的信息。

```python
CLASS ac.AiLogFastStreamRedirect2File(self,
                                      source_stream=None,
                                      redirect_nodes: Union[List, Tuple, None] = None,
                                      redirect_devices: Union[List, Tuple, None] = None,
                                      file_save_dir: str = '',
                                      append_rank_dir: bool = True,
                                      file_name: str = '')
```

- **source_stream**：需要被重定向的流。默认为 `None`，表示被重定向的流为 `sys.stderr`。

- **redirect_nodes**：该变量控制哪些计算节点的流需要被重定向。默认为 `None`，表示所有的计算节点的流都会被重定向。当值设置为`[0, 1, 2, 3]`或者`(0, 1, 2, 3)`时，表示计算节点0、1、2和3中的程序可能会重定向指定流。

- **redirect_devices**：该变量控制一个计算节点中哪些卡对应程序的流会被重定向。默认为 `None`，表示所有的卡对应程序的流都会被重定向。当值设置为`[0, 1, 2, 3]`或者`(0, 1, 2, 3)`时，表示卡0、1、2和3中的程序可能会输出日志到屏幕。

  当 `redirect_nodes` 值为 `[0, 1]`，`redirect_device` 值为 `[0, 1]` 时，表示计算节点0和1上的第0、1卡对应程序的流都会被重定向。

- **file_save_dir**：流内容保存的文件所在的文件夹。默认为 `''`，表示选择默认的文件夹进行存储，线下运行时默认文件夹为 `./log/`，线上运行时默认文件夹为 `/cache/ma-user-work/`，且不随传值发生变化。

- **append_rank_dir**：是否在 `file_save_dir` 后添加与 rank 相关的文件夹。默认为 `True`，线下运行时路径变为 `./log/rank_0`，线上同理。

- **file_name**：保存被重定向流中内容的文件的名称。默认为 `''`，表示 `mindspore.log`。

**使用样例：**

```python
import aicc_tools as ac

# 重定向 sys.stderr —— 普通用法
redirector = ac.AiLogFastStreamRedirect2File()
redirector.start()
<some code>  # 这些代码往 sys.stderr 输出的内容会被重定向到文件中，可以认为 mindspore 的日志会被重定向到文件
redirector.stop()

# 重定向 sys.stderr —— with
with ac.AiLogFastStreamRedirect2File():
    <some code>
    
# 重定向 sys.stderr —— 装饰器
@ac.AiLogFastStreamRedirect2File()
def func():
    <some code>
    
# 重定向 sys.stdout 到文件
redirector = ac.AiLogFastStreamRedirect2File(source_stream=sys.stdout)
redirector.start()
<some code>
redirector.stop()
```

# AICC Tools 简易Demo

## Demo1 Mindspore常用训练示范

```python
import os

import numpy as np
import aicc_tools as ac

from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

#注： Step1、Step3、Step4、Step5 为必须增加代码， Step2为可选代码，用户可自行初始化环境
# Step1 ---------------------> 为作业主进程加上@ac.aicc_monitor装饰器修饰，保证进程被监控保护
@ac.aicc_monitor
def main():
    obs_path = "s3://example/output"
    context_config = {"mode": 0, "device_target": "CPU", "device_id": 7}
    parallel_config = {"parallel_mode": 0, "gradients_mean": True}

    # Step2 ---------------------> init context 用于环境初始化，单卡运行。可选步骤，可自行定义环境
    rank_id, device_num = ac.context_init(seed=1, use_parallel=False,
                                          context_config=context_config,
                                          parallel_config=parallel_config)
    # Step3 ---------------------> init cfts， 初始化ctfs系统，必选步骤
    cfts = ac.CFTS(obs_path, upload_frequence=1, keep_last=False)

    # Step4 ---------------------> create aicc log system, 算法整体建议使用该logger进行打印，可有效对打印日志进行分卡分类
    logger = ac.get_logger()
    logger.info("Init log system of aicc tools.")

    # create dataset 样例简化格式，可根据自己代码需求自定义
    ds = create_dataset(data_path=cfts.get_dataset(dataset_path="dataset_path"))

    # creat model, loss, optim ....
    # 加载权重，可参考:
    # param = load_checkpoint(cfts.get_checkpoint(checkpoint_path="checkpoint_path/.ckpt"))
    # load_param_into_net(param, net)

    # 此处使用aicc tools一些常见的callback封装，自定义callback参考下面注释方式
    ckpt_cb = cfts.checkpoint_monitor(prefix="aicc",
                                      save_checkpoint_steps=100,
                                      keep_checkpoint_max=1,
                                      integrated_save=False)
    loss_cb = cfts.loss_monitor(per_print_times=1)
    summary_cb = cfts.summary_monitor(collect_freq=1)
    """
    if you use custom define callback function, please use aicc_tools`s logger print your info,
    if you need save custom file, please use cfts.get_custom_path() as your save path.
    such as:
    custom_cb1 = CustomMonitor(log=logger.info) # 不涉及文件保存,只需要实时打印观测， 注意使用aicc_tools.get_logger()进行打印
    custom_cb2 = CustomMonitor(save_dir=cfts.get_custom_path(directory="custom")) # 涉及文件保存时，需传入cfts固定的用户路径，保证文件被回传！
    """
    callback = [loss_cb, summary_cb, ckpt_cb]
    # Step5 ---------------------> 使用cfts.obs_monitor()进行自动回传
    callback.append(cfts.obs_monitor())
    
    # train model，注意cfts.obs_monitor()必须放在callbacks列表最后，以保证所有文件被回传！！！
    model.train(5, ds,callbacks=callback)

if __name__ == "__main__":
    main()
```

## Demo2 Mindspore 自定义训练示范

```python
import os

import numpy as np
import aicc_tools as ac

from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

#注： Step1、Step3、Step4、Step5 为必须增加代码， Step2为可选代码，用户可自行初始化环境
# Step1 ---------------------> 为作业主进程加上@ac.aicc_monitor装饰器修饰，保证进程被监控保护
@ac.aicc_monitor
def main():
    obs_path = "s3://example/output"
    context_config = {"mode": 0, "device_target": "CPU", "device_id": 7}
    parallel_config = {"parallel_mode": 0, "gradients_mean": True}

    # Step2 init context 用于环境初始化，可选步骤，可自行定义环境
    rank_id, device_num = ac.context_init(seed=1, use_parallel=False,
                                          context_config=context_config,
                                          parallel_config=parallel_config)
    # Step3 ---------------------> init cfts， 初始化ctfs系统，必选步骤
    cfts = ac.CFTS(obs_path, upload_frequence=1, keep_last=False)

    # Step4 ---------------------> create aicc log system, 算法整体建议使用该logger进行打印，可有效对打印日志进行分卡分类
    logger = ac.get_logger()
    logger.info("Init log system of aicc tools.")

    # create dataset 样例简化格式，可根据自己代码需求自定义
    ds = create_dataset(data_path=cfts.get_dataset(dataset_path="dataset_path"))

    # creat model, loss, optim ....
    # 加载权重，可参考:
    # param = load_checkpoint(cfts.get_checkpoint(checkpoint_path="checkpoint_path/.ckpt"))
    # load_param_into_net(param, net)
    
    for epoch in range(5):
        # if you want print information, please use aicc_tools.get_logger().
        logger.info("Train starting.")
        for data in ds.create_tuple_iterator():
            # define yourself training process.
            .....
        # if you want save checkpoint file , you can follow as: 请使用cfts中默认路径，以保证文件被回传
        save_checkpoint(
            net,ckpt_file_name=cfts.get_custom_path(directory="ckpt", file_name="net_{}.ckpt".format(epoch)))

        # Step5 ---------------------> 程序最后执行回传命令,自动回传所有被保存的文件和日志, 每个epoch回传一次
        cfts.send2obs()

if __name__ == "__main__":
    main()
```

