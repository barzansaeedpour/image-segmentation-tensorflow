# Image Segmentation Tensorflow



## Table of content

- [Image Segmentation](#image-segmentation)
- [( Semantic vs Instance ) Segmentation](#semantic-vs-instance-segmentation )
- [Image Segmentation Basic Architecture](#image-segmentation-basic-architecture)
- [Popular Architectures](#popular-architectures)
    - [well-known networks based on FCN](#well-known-networks-based-on-fcn)
        - [Segnet](#segnet)


## Image Segmentation

Image segmentation is a computer vision task that involves dividing an image into meaningful and relevant regions or segments. The goal is to group together pixels or regions in the image that have similar characteristics such as color, texture, or intensity, while differentiating them from the surrounding areas.


| Original image                               | Segmentation Output                          |
| -------------------------------------------  | -------------------------------------------- |
| ![Image 1](./files/image1.png)               | ![Image 2](./files/image2.png)               |


## ( Semantic vs Instance ) Segmentation 

- Semantic segmentation focuses on labeling each pixel with a class label, providing a high-level understanding of the image.
- Instance segmentation aims to not only assign class labels but also differentiate individual instances of objects, providing a more detailed and precise segmentation.


| Semantic segmentation                         | Instance segmentation                          |
| -------------------------------------------   | -------------------------------------------- |
| ![Image 1](./files/semantic-segmentation.jpeg)| ![Image 2](./files/instance-segmentation.jpeg)               |

## Image Segmentation Basic Architecture

Let's explore the basic architecture for a model that can be used for segmentation. 
 
The high level architecture for an image segmentation algorithm is an encoder-decoder one. 

![Image 1](./files/architecture.png)


 - **input image**: It works by taking an input image which has a set of dimensions, such as 224 by 224 by 3 for a color red, green, blue one. 
 - **encoder**: The image is processed with an encoder. An encoder is a feature extractor. And for image processing, it's typical to use a CNN to extract features. The encoder extracts features from the image into a feature map. The earlier layers extract low level features such as lines and these lower level features are successfully aggregated into higher level features such as eyes and ears. The aggregation of successful higher level features is done with the help of downsampling. 
 - **decoder**: The image segmentation architecture will take the downsampled feature map and feed it to a decoder. The decoders task is to take the features that were extracted by the encoder and work on producing the models output or prediction. The decoder is also a convolutional neural network. The decoder assigns intermediate class labels to each pixel of the feature map, and then up samples the image to slowly add back the fine grained details of the original image. The decoder then assigns more fine grained intermediate class labels to the up samples pixels and repeats this process until the images up sampled back to its original input dimensions. The final predicted image also has the final class labels assigned to each pixel.
- **pixel wise labeled map**: This then gives a pixel wise labeled map. In this example, the pixel wise labeled map will be the size of the original image. For example, 224 by 224, with the third dimension being the number of classes. So each slice of 224 by 224 will be the mappings of the pixels for that particular class.

- **Algorithm**

    - **Encoder**
        - CNN without fully connected layes
        - Aggregates low level features to high level features

    - **Decoder**: 
        - Replaces fully connected layers in a CNN
        - Up samples image to original size to generate a pixel mask

## Popular Architectures

- Fully Convolutional Neural Networks (FCNs)
    
    - properties:
        - Replace the fully connected layers with convolutional layes
        - Earlier conv layers: Feature extraction and down sampling
        - Later conv layers: up sample and pixel-wise labelmap
    - FCN architecture:
        
        This is an illustration of the architecture from the original paper. It shows an example of how filters are learned in the usual way through forward inference and backpropagation. At the end is a pixel-wise prediction layer that will create the segmentation map.
        !["fully-convolutional-neural-networks"](./files/fully-convolutional-neural-networks.png)
        
        The decoder has a number of options:

        <img width="60%" src="./files/comparison-of-different-fcns.png"/>

        - Fully convolutional neural networks, encoders are feature extractors like the feature extracting layers using object detection models. So you can reuse the layers of well-known object detection models as the encoder of the fully connected network. For example, VGG16, ResNet 50, or MobileNet, have pre-trained feature extraction layers that you can use.
        - The decoder part of the FCN is usually called FCN-32, FCN-16 or FCN-8 with a number denotes the stride size during upsampling. You may recall that the stride in a convolutional layer determines how many pixels to shift the sliding window as it traverses the image. The smaller the stride, the more detailed the processing. The difference between the decoder architectures ends up effectively being the resolution of the final pixel map. You can see that here as the resolution improves, as the strike decreases from 32-16 and then to eight, and eight is the closest to the ground truth
    - ### well-known networks based on FCN:
        - #### SegNet

            <img width="70%" src="./files/segnet.png"/>
            
            SegNet Is very similar to the fully connected CNN with a notable optimization. That is that the encoder layers are symmetric with the decoder layers. They like mirror images of each other with the same number of layers and the same arrangement of those layers. For example, for each pooling layer that downsampled in the encoder, there was an upsampling layer and the decoder section. For example, in this architecture, the first segment has two convolutional layers, followed by a pooling layer. The last segment is a mirror image of this with an upsampling layer followed by two convolutional layers. The same symmetry is found in the second layer and the second-to-last one, and so on for the rest of the image
        - #### UNet
        - #### PSPNet
        - #### Mask-RCNN