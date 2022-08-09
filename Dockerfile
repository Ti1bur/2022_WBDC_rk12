FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-trt8.2.5

WORKDIR /opt/ml/wxcode

COPY ./ ./

