# Real-Noise-Image-Adjustment-Networks-for-Saliency-aware-Stylistic-Color-Retouch
## Abstract

Automatic Image Adjustment (AIA) method mainly aims to realize stylistic color retouch in images. Recent years have witnessed the unprecedented success in the learning-based AIA methods, especially Convolutional Neural Networks (CNNs). However, existing AIA methods usually handle the images without real noise from ideal scenarios, resulting in poor retouch performance when processing the real noise images. Furthermore, these AIA methods lack attentive capability in learning salient areas to perform the stylistic color retouch as human artists do. To address these problems, we first re-models adjustment task for real noise images to remove the real noise. Then, we further propose the Real Noise Image Adjustment Networks (RNIA-Nets) using adaptive denoise and saliency-aware stylistic color retouch. Specifically, an adaptive denoise mechanism pertinently predicts denoise kernel for various real noise images. The saliency-aware stylistic color retouch predicts visual salient areas to learn stylistic color mapping through a proposed Multi-faced Attention (MFA) module. Eventually, to equitably verify the effectiveness of the proposed RNIA-Nets, a new challenging benchmark dataset collected from real noise images is established. Extensive experimental results demonstrate that the proposed method can achieve favorable results on real noise image adjustment, providing a highly effective solution to the practical AIA applications.

### The dataset and code will be released as soon as possible.

<img src="https://github.com/JiangBoCS/Real-Noise-Image-Adjustment-Networks-for-Saliency-aware-Stylistic-Color-Retouch/blob/main/framework.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>The overall framework of RNIA-Nets. RNIA-Nets has two branch structures.</p></center>

## Multi-Faced Attention (MFA) Block with Feedback Mechanism Structure.
<img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Multi-Faced%20Attention%20(MFA)%20Block%20with%20Feedback%20Mechanism%20Structure.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>Multi-Faced Attention (MFA) Block.</p></center>

## Experimental visual effect results
<img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Noisy%20image_10_3-RNIA-Nets%20(Ours)_10_3.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>Style 1.</p></center>

 <img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Noisy%20image_11_2-RNIA-Nets%20(Ours)_11_2.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
 <center><p>Style 2.</p></center>

<img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Noisy%20image_3_2-RNIA-Nets%20(Ours)_3_2.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
 <center><p>Style 3.</p></center>
 
 <img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Noisy%20image_2_8-RNIA-Nets%20(Ours)_2_8.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
 <center><p>Style 4.</p></center>
 
  <img src="https://github.com/JiangBoCS/RNIA-Nets/blob/main/Noisy%20image_18_7-RNIA-Nets%20(Ours)_18_7.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
 <center><p>Style 5.</p></center>
 

 
 
 

