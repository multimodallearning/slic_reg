#pytorch 1.9 code

as detailed in the main readme our proposed SLIC_REG requires a baseline registration that nonlinearly transforms a set of supervoxels from a reference/template scan to all training scans. We then train a fully-convolutional network (3D DeepLab, MobileNetV2 + ASPP) to predict the same supervoxels for unseen scans. Those predictions are used within an Adam based alignment model.  
