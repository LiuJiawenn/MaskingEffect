these scripts implemented the spacial-temporal masking effect proposed by Hu et al. in the papers:

[1] Hu, S., Jin, L., Wang, H., Zhang, Y., Kwong, S., & Kuo, C. C. J. (2015). Compressed image quality metric based on 
perceptually weighted distortion. IEEE Transactions on Image Processing, 24(12), 5594-5608.

[2] Hu, S., Jin, L., Wang, H., Zhang, Y., Kwong, S., & Kuo, C. C. J. (2016). Objective video quality assessment 
based on perceptually weighted mean squared error. IEEE Transactions on Circuits and Systems for Video Technology, 27(9), 1844-1855.

# for spacial masking
i tested with 9*9 block and it runs well, but the cupy version is slower. 
it's about 20s for cpu to run 1 frame with resolution 1080×1980

<img src="https://github.com/LiuJiawenn/MaskingEffect/assets/124513316/2ce389d1-616b-427d-b8d9-5aec37d2c14a" width="180" height="320">


# for temporal masking
the biggest block that i can test is 90×160 with my small RAM. The cupy version is faster.
it's about 20s for gpu to run 1 frame masking response with 30frames with resolution 1080×1980

<img src="https://github.com/LiuJiawenn/MaskingEffect/assets/124513316/32df1bf0-4525-41c3-94d0-08d4671feea8" width="180" height="320">
