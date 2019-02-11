require 'torch'
require 'nn'
require 'nngraph'

nfeats = 2
width = 22*50
height = 1
ninputs = nfeats*width*height
--nkernels = {320,480,960,100}
chrom_width=width/22;
nkernels = {160,240,480,22}

--determin which chromosome to change
Chrom_Net = nn.Sequential()


Chrom_Net:add(nn.SpatialConvolution(nfeats, nkernels[1], 1, 7, 1, 1, 0,3))
Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--Chrom_Net:add(nn.Dropout(0.2))

Chrom_Net:add(nn.SpatialConvolution(nkernels[1], nkernels[2], 1, 8, 1, 1, 0,3))
Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.SpatialMaxPooling(1,4,1,4,0,2))
--Chrom_Net:add(nn.Dropout(0.2))

Chrom_Net:add(nn.SpatialConvolution(nkernels[2], nkernels[3], 1, 8, 1, 1, 0,3))
Chrom_Net:add(nn.Threshold(0, 1e-6))
---Chrom_Net:add(nn.Dropout(0.5))

nchannel = math.floor((math.floor((width)/4.0))/4.0)
Chrom_Net:add(nn.Reshape(nkernels[3]*nchannel))
Chrom_Net:add(nn.Linear(nkernels[3]*nchannel, nkernels[4]))
Chrom_Net:add(nn.Threshold(0, 1e-6))
Chrom_Net:add(nn.Linear(nkernels[4] ,nkernels[4]))
--Chrom_Net:add(nn.SoftMax())
Chrom_Net:add(nn.Exp())

Chrom_Model=Chrom_Net;

--determine the starting position and type of CNV
CNV_i1 =-nn.SpatialConvolution(nfeats,nkernels[1]/2, 1, 7, 1, 1, 0,3)
CNV_h1 =CNV_i1-nn.Threshold(0, 1e-6)
CNV_h1 =CNV_h1-nn.SpatialMaxPooling(1,20,1,20,0,2)

CNV_i2 =-nn.SpatialConvolution(nfeats,nkernels[1]/2, 1, 7, 1, 1, 0,3)
CNV_h2 =CNV_i2-nn.Threshold(0, 1e-6)

CNV_Net={CNV_h1,CNV_h2}-nn.JoinTable(2,3)

CNV_Net =CNV_Net-nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 8, 1, 1, 0,3)
CNV_Net =CNV_Net-nn.Threshold(0, 1e-6)
CNV_Net =CNV_Net-nn.SpatialMaxPooling(1,4,1,4,0,2)


CNV_Net =CNV_Net-nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 8, 1, 1, 0,3)
CNV_Net =CNV_Net-nn.Threshold(0, 1e-6)

nchannel = math.floor((math.floor((width)/20)+chrom_width)/4)
CNV_Net =CNV_Net-nn.Reshape(nkernels[3]/2*nchannel)
CNV_Net =CNV_Net-nn.Linear(nkernels[3]/2*nchannel,4*chrom_width)
--CNV_Net =CNV_Net-nn.SoftMax();

CNV_Model=nn.gModule({CNV_i1,CNV_i2},{CNV_Net});

--determin the end position
End_Point_i1=-nn.Identity()
End_Point_i2=-nn.Identity()
End_Point_Net={End_Point_i1,End_Point_i2}-nn.JoinTable(1,3)
End_Point_Net=End_Point_Net-nn.SpatialConvolution(2*nfeats, nkernels[1]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)
--End_Point_Net =End_Point_Net-nn.SpatialMaxPooling(1,4,1,4,0,4-width%4)


End_Point_Net =End_Point_Net-nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)
--CNV_Net =CNV_Net-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)
End_Point_Net =End_Point_Net-nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 7, 1, 1, 0,3)
End_Point_Net =End_Point_Net-nn.Threshold(0, 1e-6)

End_Point_Net =End_Point_Net-nn.Reshape(nkernels[3]/2*chrom_width)
End_Point_Net =End_Point_Net-nn.Linear(nkernels[3]/2*chrom_width,chrom_width)
--End_Point_Net= End_Point_Net-nn.SoftMax()

End_Point_Model=nn.gModule({End_Point_i1,End_Point_i2},{End_Point_Net});
