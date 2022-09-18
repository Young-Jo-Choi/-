# FCN (Fully Convolutional Network) for Semantic Segmentation

(수정)
Architecture
- 어떤 사이즈의 input image라도 입력받을 수 있도록
- Upsampling - Deconvolution
- Skip Connection
Pixel-wise semantic segmentation


1. convolutionalization
2. Deconvolution(Upsampling)
3. Skip architecture

# Introduction
기존 CNN - encoder + fully connected 
(figure1_3 첨부)

fc layer의 한계 - 위치 정보의 손실, 입력 이미지 크기가 고정됨
어떤 방법을 사용했는지 등등
## 기존 연구와의 차이점은 무엇인지

출력층을 fully connected layer --> Convnet으로 변경(모양은?)
위치정보가 매우 중요
(개,고양이 사진 첨부)

(fc layer->convnet 사진 첨부)

convnet으로 변하는 출력층의 shape 설명
FCN의 출력은 coarse

Deconvolution
연산과정은 d2l 책 참고

skip connection
figure3 첨부 
의미에 대해 설명

결과
figure4, result_table

