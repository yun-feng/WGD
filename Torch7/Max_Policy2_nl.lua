require 'torch'
require 'nn'
require 'nngraph'
require 'math'

nfeats = 2
width = 22*50
height = 1
ninputs = nfeats*width*height
--nkernels = {320,480,960,100}
chrom_width=width/22;
nkernels = {160,240,480,960,22*2}

--determin which chromosome to change
Chrom_Net = nn.Sequential()

Chrom_Net:add(nn.AddConstant(-1))
Chrom_Net:add(nn.Reshape(1,2*1100,1))
Chrom_Net:add(nn.Reshape(1,44,50))
Chrom_Net:add(nn.SpatialConvolution(nfeats/2, nkernels[1]/2, 7, 1, 1, 1, 3,0))
Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.SpatialMaxPooling(3,1,3,1,1,0))
--Chrom_Net:add(nn.Dropout(0.2))

Chrom_Net:add(nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 8, 1, 1, 1, 3,0))
Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.SpatialMaxPooling(2,1,2,1,1,0))
--Chrom_Net:add(nn.Dropout(0.2))

Chrom_Net:add(nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 8, 1, 1, 1, 3,0))
Chrom_Net:add(nn.Threshold(0, 1e-6))
---Chrom_Net:add(nn.Dropout(0.5))
Chrom_Net:add(nn.SpatialMaxPooling(2,1,2,1,1,0))

Chrom_Net:add(nn.SpatialConvolution(nkernels[3]/2, nkernels[4]/2, 8, 1, 1, 1, 3,0))
--Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.Threshold())

Chrom_Net:add(nn.Replicate(2))
Chrom_Net:add(nn.SplitTable(1))
p0=nn.ParallelTable()
p0:add(nn.Sequential():add(nn.SpatialConvolution(nkernels[4]/2, 5, 4, 1, 1, 1, 0,0)):add(nn.Sum(3,4)):add(nn.Sigmoid()):add(nn.Replicate(44,3)):add(nn.Reshape(5,44)))

p1=nn.ParallelTable()
                                                                     
p1:add(nn.Sequential():add(nn.SpatialConvolution(nkernels[4]/2, 5, 4, 1, 1, 1, 0,0)):add(nn.Reshape(5,44)))
p1:add(nn.Sequential():add(nn.SpatialConvolution(nkernels[4]/2, 5, 4, 1, 1, 1, 0,0)):add(nn.Sum(3,4)):add(nn.Replicate(44,3)):add(nn.Reshape(5,44)))

p0:add(nn.Sequential():add(nn.Replicate(2)):add(nn.SplitTable(1)):add(p1):add(nn.CAddTable()):add(nn.ELU()))
Chrom_Net:add(p0)
Chrom_Net:add(nn.CMulTable())
Chrom_Net:add(nn.Sum(2))

--nchannel=Chrom_Net:forward(torch.ones(2,1100,1)):size(2)
--nchannel = math.floor((math.floor((width)/4.0))/4.0)
--Chrom_Net:add(nn.Reshape(nkernels[4]/2*nchannel))
--Chrom_Net:add(nn.Linear(nkernels[4]/2*nchannel, nkernels[5]))
--Chrom_Net:add(nn.Sigmoid())
--Chrom_Net:add(nn.Replicate(2,2))
--Chrom_Net:add(nn.SplitTable(2))

--p2=nn.ParallelTable()
--p2:add(nn.Mul())
--p2:add(nn.Sequential():add(nn.Mean(2)):add(nn.Mul()):add(nn.Replicate(44,2)):add(nn.Reshape(44)))
--Chrom_Net:add(p2)
--Chrom_Net:add(nn.CAddTable())
--Chrom_Net:add(nn.Linear(nkernels[5] ,nkernels[5]))
--Chrom_Net:add(nn.AddConstant(10))
--Chrom_Net:add(nn.SoftMax())
--Chrom_Net:add(nn.Exp())
--Chrom_Net:add(nn.ELU(1))
--Chrom_Net:add(nn.AddConstant(400))
--Chrom_Net:add(nn.AddConstant(-math.log(5e-5*0.99)))

Chrom_Model=Chrom_Net;

--determine the starting position and type of CNV
CNV_i1 =-nn.SpatialConvolution(nfeats,nkernels[1]/2, 1, 7, 1, 1, 0,3)
CNV_h1 =CNV_i1-nn.Threshold(0, 1e-6)
CNV_h1 =CNV_h1-nn.SpatialMaxPooling(1,20,1,20,0,2)

CNV_i2 =-nn.SpatialConvolution(1,nkernels[1]/2, 1, 7, 1, 1, 0,3)
CNV_h2 =CNV_i2-nn.Threshold(0, 1e-6)

CNV_Net={CNV_h1,CNV_h2}-nn.JoinTable(2,3)

CNV_Net =CNV_Net-nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 8, 1, 1, 0,3)
CNV_Net =CNV_Net-nn.Threshold(0, 1e-6)
CNV_Net =CNV_Net-nn.SpatialMaxPooling(1,4,1,4,0,2)


CNV_Net =CNV_Net-nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 8, 1, 1, 0,3)
CNV_Net =CNV_Net-nn.Threshold(0, 1e-6)

nchannel = math.floor((math.floor((width)/20)+chrom_width)/4)
CNV_Net =CNV_Net-nn.Reshape(nkernels[3]/2*nchannel)
CNV_Net =CNV_Net-nn.Linear(nkernels[3]/2*nchannel,2*chrom_width-1)
--CNV_Net =CNV_Net-nn.SoftMax();
--CNV_Net=CNV_Net-nn.Sigmoid()

CNV_Model=nn.gModule({CNV_i1,CNV_i2},{CNV_Net});

--determin the end position
End_Point_i1=-nn.Identity()
End_Point_i2=-nn.Identity()
End_Point_Net={End_Point_i1,End_Point_i2}-nn.JoinTable(1,3)
End_Point_Net=End_Point_Net-nn.SpatialConvolution(1*nfeats, nkernels[1]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)
--End_Point_Net =End_Point_Net-nn.SpatialMaxPooling(1,4,1,4,0,4-width%4)


End_Point_Net =End_Point_Net-nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)
--CNV_Net =CNV_Net-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)
End_Point_Net =End_Point_Net-nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)

End_Point_Net =End_Point_Net-nn.Reshape(nkernels[3]/2*chrom_width)
End_Point_Net =End_Point_Net-nn.Linear(nkernels[3]/2*chrom_width,chrom_width-1)
--End_Point_Net= End_Point_Net-nn.SoftMax()
--End_Point_Net=End_Point_Net-nn.Sigmoid()
End_Point_Model=nn.gModule({End_Point_i1,End_Point_i2},{End_Point_Net});
