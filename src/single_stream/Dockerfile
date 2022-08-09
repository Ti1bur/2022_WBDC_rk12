FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-trt8.2.5

WORKDIR /opt/ml/wxcode

COPY ./opensource_models ./opensource_models
# COPY ./save/fold5/model1.bin ./save/fold5/model1.bin
# COPY ./save/fold5/model2.bin ./save/fold5/model2.bin
# COPY ./save/fold5/model3.bin ./save/fold5/model3.bin
# COPY ./save/fold5/model4.bin ./save/fold5/model4.bin
COPY ./save/model_epoch_2_mean_f1_0.7217.bin ./save/model_epoch_2_mean_f1_0.7217.bin
COPY ./TensorRT ./TensorRT

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

COPY ./*.py ./
COPY ./start.sh ./

CMD sh -c "sh start.sh"
