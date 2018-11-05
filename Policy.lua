require 'torch'
require 'nn'

nfeats = 2
width = trainData.data:size(3)
height = 1
ninputs = nfeats*width*height
nkernels = {160,240,480,22+2}


ChromNet = nn.Sequential()


ChromNet:add(nn.SpatialConvolutionMM(nfeats, nkernels[1], 1, 8, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.SpatialMaxPooling(1,4,1,4,0,4-width%4))
--ChromNet:add(nn.Dropout(0.2))

ChromNet:add(nn.SpatialConvolutionMM(nkernels[1], nkernels[2], 1, 8, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.SpatialMaxPooling(1,4,1,4,0,4-math.ceil(width/4)%4))
--ChromNet:add(nn.Dropout(0.2))

ChromNet:add(nn.SpatialConvolutionMM(nkernels[2], nkernels[3], 1, 8, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
--ChromNet:add(nn.Dropout(0.5))

nchannel = math.ceil((math.ceil((width)/4.0))/4.0)
ChromNet:add(nn.Reshape(nkernels[3]*nchannel))
ChromNet:add(nn.Linear(nkernels[3]*nchannel, nkernels[4]))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.Linear(nkernels[4] ,nkernels[4]))
ChromNet:add(nn.SoftMax())

require "nngraph";

DoubleValue =-nn.SpatialConvolutionMM(nfeats,nkernels[1]/4, 1, 8, 1, 1, 0,3)
DoubleValue =DoubleValue-nn.Threshold(0, 1e-6)
DoubleValue =DoubleValue-nn.SpatialMaxPooling(1,20,1,20,0,20-width%20)


DoubleValue =DoubleValue-nn.SpatialConvolutionMM(nkernels[1]/4, nkernels[2]/4, 1, 8, 1, 1, 0,3)
DoubleValue =DoubleValue-nn.Threshold(0, 1e-6)
DoubleValue =DoubleValue-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)


DoubleValue =DoubleValue-nn.SpatialConvolutionMM(nkernels[2]/4, nkernels[3]/4, 1, 8, 1, 1, 0,3)
DoubleValue =DoubleValue-nn.Threshold(0, 1e-6)

nchannel = math.ceil((math.ceil((width)/20))/5)
DoubleValue =DoubleValue-nn.Reshape(nkernels[3]/4*nchannel)
DoubleValue =DoubleValue-nn.linear(nkernels[3]/4*nchannel,5*50)

FocusNet=-nn.SpatialConvolutionMM(nfeats, nkernels[1], 1, 8, 1, 1, 0,3)
FocusNet =FocusNet-nn.Threshold(0, 1e-6)
--FocusNet =FocusNet-nn.SpatialMaxPooling(1,4,1,4,0,4-width%4)


FocusNet =FocusNet-nn.SpatialConvolutionMM(nkernels[1], nkernels[2], 1, 8, 1, 1, 0,3)
FocusNet =FocusNet-nn.Threshold(0, 1e-6)
--DoubleValue =DoubleValue-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)


FocusNet =FocusNet-nn.Reshape(nkernels[2]*50)
FocusNet =FocusNet-nn.linear(nkernels[3]/4*50,5*50)

UpperPolicyNet=nn.CAddTable()({FocusNet,DoubleValue})
UpperPolicyNet=UpperPolicyNet-nn.SoftMax()
