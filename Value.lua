require 'torch'
require 'nn'

require 'math'

--seq as nfeats*width*1




nfeats = 2
width = 22*50
height = 1
ninputs = nfeats*width*height
nkernels = {320,480,960,100}


ValueNet = nn.Sequential()


ValueNet:add(nn.SpatialConvolution(nfeats, nkernels[1], 1, 7, 1, 1, 0,3))
ValueNet:add(nn.Threshold(0, 1e-6))
ValueNet:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--ValueNet:add(nn.Dropout(0.2))

ValueNet:add(nn.SpatialConvolutionMM(nkernels[1], nkernels[2], 1, 8, 1, 1, 0,3))
ValueNet:add(nn.Threshold(0, 1e-6))
ValueNet:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--ValueNet:add(nn.Dropout(0.2))

ValueNet:add(nn.SpatialConvolutionMM(nkernels[2], nkernels[3], 1, 8, 1, 1, 0,3))
ValueNet:add(nn.Threshold(0, 1e-6))
--ValueNet:add(nn.Dropout(0.5))

nchannel = math.floor((math.floor((width)/4.0))/4.0)
ValueNet:add(nn.Reshape(nkernels[3]*nchannel))
ValueNet:add(nn.Linear(nkernels[3]*nchannel, nkernels[4]))
ValueNet:add(nn.Tanh())
--ValueNet:add(nn.Linear(nkernels[4], nkernels[4]))
ValueNet:add(nn.Linear(nkernels[4] , 1))
ValueNet:add(nn.Exp())

ValueNet_eval=ValueNet:clone()
