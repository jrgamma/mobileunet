# MobilNetV3-UNet Autoencoder
A simple Pytorch implementation of the [UNet architecture](https://arxiv.org/abs/1505.04597) based on the lightweight [MobileNetV3](https://arxiv.org/abs/1905.02244) encoder, with the decoder modules mirroring the ones of the encoder.

### Architecture Variants
Similar to the official [`torchvision` implementation](https://docs.pytorch.org/vision/main/models/mobilenetv3.html), the following variants are available, with the total number of trainable parameters shown as follows:
| Architecture | Encoder | Decoder* | Total |
|--------------|---------|----------|-------|
| Small        | 927K    | 959K     | 1.89M |
| Large        | 2.97M   | 3.10M    | 6.07M |

*assuming single channel output decoder.

## Usage
The only dependency for `mobileunet` is `Pytorch >= 2.0`.

To instantiate the model,
```python
from mobileunet import mobileunet

model_small = mobileunet.mobileunet_small()
model_large = mobileunet.mobileunet_large()
```
