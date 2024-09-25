# enginner_vision
工程自动兑换识别工程文件

![识别效果图](./img/box.jpg)

# 安装OpenVino（Running Time 和 Develop Toolkit都要安装）

```SH
# Running Time
sudo mkdir /opt/intel

cd /tmp

curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz

tar -xf openvino_2023.0.0.tgz

sudo mv l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0

cd /opt/intel/openvino_2023.0.0
sudo -E ./install_dependencies/install_openvino_dependencies.sh

cd /opt/intel
sudo ln -s openvino_2023.0.0 openvino_2023

echo 'source /opt/intel/openvino_2023/setupvars.sh' >> ~/.bashrc

# Develop Toolkit

python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openvino-dev==2023.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

---

## 使用OpenVino将onnx模型转换成ir模型

```SH
mo --input_model onnx模型的文件路径 --mean_values [103.53,116.28,123.675] --scale_values [57.375,57.12,58.395] --output output --output_dir 输出到的文件夹
```

会生成.bin和.xml文件，使用时将两个文件放在同一个文件夹里面
