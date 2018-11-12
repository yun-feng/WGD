require 'torch'
require 'nn'
require 'nngraph'

--seq as nfeats*width*1


nfeats = 2
width = trainData.data:size(3)
chrom_width=width/22

height = 1
ninputs = nfeats*width*height
nkernels = {160,240,480,22+2}


--determin which chromosome to change
ChromNet = nn.Sequential()


ChromNet:add(nn.SpatialConvolution(nfeats, nkernels[1], 1, 7, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--ChromNet:add(nn.Dropout(0.2))

ChromNet:add(nn.SpatialConvolution(nkernels[1], nkernels[2], 1, 8, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--ChromNet:add(nn.Dropout(0.2))

ChromNet:add(nn.SpatialConvolution(nkernels[2], nkernels[3], 1, 8, 1, 1, 0,3))
ChromNet:add(nn.Threshold(0, 1e-6))
---ChromNet:add(nn.Dropout(0.5))

nchannel = math.floor((math.floor((width)/4.0))/4.0)
ChromNet:add(nn.Reshape(nkernels[3]*nchannel))
ChromNet:add(nn.Linear(nkernels[3]*nchannel, nkernels[4]))
ChromNet:add(nn.Threshold(0, 1e-6))
ChromNet:add(nn.Linear(nkernels[4] ,nkernels[4]))
ChromNet:add(nn.SoftMax())


--determine the starting position and type of CNV
Double_i1 =-nn.SpatialConvolution(nfeats,nkernels[1]/4, 1, 7, 1, 1, 0,3)
Double_h1 =Double_i1-nn.Threshold(0, 1e-6)
Double_h1 =Double_h1-nn.SpatialMaxPooling(1,20,1,20,0,2)

Double_i2 =-nn.SpatialConvolution(nfeats,nkernels[1]/4, 1, 7, 1, 1, 0,3)
Double_h2 =Double_i2-nn.Threshold(0, 1e-6)

DoubleValue={Double_h1,Double_h2}-nn.JoinTable(2,3)

DoubleValue =DoubleValue-nn.SpatialConvolution(nkernels[1]/4, nkernels[2]/4, 1, 8, 1, 1, 0,3)
DoubleValue =DoubleValue-nn.Threshold(0, 1e-6)
DoubleValue =DoubleValue-nn.SpatialMaxPooling(1,4,1,4,0,2)


DoubleValue =DoubleValue-nn.SpatialConvolution(nkernels[2]/4, nkernels[3]/4, 1, 8, 1, 1, 0,3)
DoubleValue =DoubleValue-nn.Threshold(0, 1e-6)

nchannel = math.floor((math.floor((width)/20)+chrom_width)/4)
DoubleValue =DoubleValue-nn.Reshape(nkernels[3]/4*nchannel)
DoubleValue =DoubleValue-nn.linear(nkernels[3]/4*nchannel,5*chrom_width)


--determin the end position
Focus_i1=-nn.Identity()
Focus_i2=-nn.Identity()
FocusNet={Focus_i1,Focus_i2}-nn.JoinTable(1,3)
FocusNet=FocusNet-nn.SpatialConvolution(2*nfeats, nkernels[1]/4, 1, 7, 1, 1, 0,3)
FocusNet =FocusNet-nn.Threshold(0, 1e-6)
--FocusNet =FocusNet-nn.SpatialMaxPooling(1,4,1,4,0,4-width%4)


FocusNet =FocusNet-nn.SpatialConvolution(nkernels[1], nkernels[2]/4, 1, 7, 1, 1, 0,3)
FocusNet =FocusNet-nn.Threshold(0, 1e-6)
--DoubleValue =DoubleValue-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)


FocusNet =FocusNet-nn.Reshape(nkernels[2]/4*ChromNet)
FocusNet =FocusNet-nn.linear(nkernels[2]/4,chrom_width)

--UpperPolicyNet=nn.CAddTable()({FocusNet,DoubleValue})
--UpperPolicyNet=UpperPolicyNet-nn.SoftMax()
