# TimeEmb

## 开始使用

### 1、环境要求

开始之前，请确保您的系统上已安装 Conda，并按照以下步骤配置环境：

```
conda create -n TimeEmb python=3.8
conda activate TimeEmb
pip install -r requirements.txt
```

### 2、下载数据

TimeEmb 所需的所有数据集都可以从 [[Google Drive]](https://drive.google.com/drive/folders/1dfnzGafiaxo6BUsCMZbmlE0N6G5_yFqK?usp=sharing) 获取。
创建一个名为 `./dataset` 的独立文件夹，并将所有 CSV 文件放入该目录中。
**注意**：请直接将 CSV 文件放入该目录中，例如 `./dataset/ETTh1.csv`。

### 3、训练示例

您可以通过运行提供的脚本命令轻松复现论文中的结果。例如，要复现主要结果，请执行以下命令：

```
sh run_main.sh
```