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



Chrom_WGD_res=nn.Sequential()
Chrom_WGD_res:add(nn.Replicate(2))
Chrom_WGD_res:add(nn.SplitTable(1))
p=nn.ParallelTable()
p1=nn.Sequential()
p1:add(nn.Reshape(2200))
p1:add(nn.Mean(1,1))
p1:add(nn.AddConstant(-1.5))
p1:add(nn.MulConstant(20))

p2=nn.Sequential()
p2:add(nn.Reshape(44,50,1))
p2:add(nn.Transpose({2,4}))
p2:add(nn.SpatialConvolution(1, nkernels[1]/4, 1, 3, 1, 1, 0,1))
p2:add(nn.Threshold(0, 1e-6))
p2:add(nn.SpatialMaxPooling(1,5,1,5,0,0))

p2:add(nn.SpatialConvolution(nkernels[1]/4, nkernels[2]/4, 1, 3, 1, 1, 0,1))
p2:add(nn.Threshold(0, 1e-6))
p2:add(nn.SpatialMaxPooling(1,2,1,2,0,0))

p2:add(nn.SpatialConvolution(nkernels[2]/4, nkernels[3]/4, 1, 5, 1, 1, 0,0))
p2:add(nn.Threshold(0, 1e-6))
p2:add(nn.Sum(4))
p2:add(nn.Reshape(nkernels[3]/4))


--p2:add(nn.Reshape(30*5))
p2:add(nn.Linear(nkernels[3]/4,1))
--p2:add(nn.MulConstant(2e-2))

p:add(p1)
p:add(p2)
Chrom_WGD_res:add(p)
Chrom_WGD_res:add(nn.CAddTable())
Chrom_WGD_res:add(nn.Sigmoid())
Chrom_WGD_res:add(nn.Replicate(2))
Chrom_WGD_res:add(nn.SplitTable(1))
switch=nn.ParallelTable()
switch:add(nn.Sequential():add(nn.MulConstant(-1)):add(nn.AddConstant(1)):add(nn.Reshape(1)))
switch:add(nn.Sequential():add(nn.Identity()):add(nn.Reshape(1)))
Chrom_WGD_res:add(switch)
Chrom_WGD_res:add(nn.JoinTable(2,2))
--Chrom_WGD_res:add(nn.Replicate(44,2))


Chrom_Net_noWGD= nn.Sequential()
Chrom_Net_noWGD:add(nn.Reshape(44,50,1))
Chrom_Net_noWGD:add(nn.Transpose({2,4}))
Chrom_Net_noWGD:add(nn.SpatialConvolution(1, nkernels[1]/2, 1, 5, 1, 1, 0,2))
Chrom_Net_noWGD:add(nn.Threshold(0, 1e-6))
Chrom_Net_noWGD:add(nn.SpatialMaxPooling(1,3,1,3,0,1))

Chrom_Net_noWGD:add(nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 3, 1, 1, 0,1))
Chrom_Net_noWGD:add(nn.Threshold(0, 1e-6))
Chrom_Net_noWGD:add(nn.SpatialMaxPooling(1,2,1,2,0,1))

Chrom_Net_noWGD:add(nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/3, 1, 3, 1, 1, 0,1))
Chrom_Net_noWGD:add(nn.Threshold(0, 1e-6))
Chrom_Net_noWGD:add(nn.SpatialMaxPooling(1,2,1,2,0,1))

Chrom_Net_noWGD:add(nn.SpatialConvolution(nkernels[3]/3, 1, 1, 5, 1, 1, 0,0))
Chrom_Net_noWGD:add(nn.ELU(0.25))
Chrom_Net_noWGD:add(nn.AddConstant(0.25))

Chrom_Net_noWGD:add(nn.Reshape(44))
--Chrom_Net_noWGD:add(nn.Sum(2))
--Chrom_Net_noWGD:add(nn.Reshape(1))

Chrom_Net_WGD=Chrom_Net_noWGD:clone()


Chrom_i1=-nn.Identity()
Chrom_i2=-nn.AddConstant(-1)
Chrom_i3=-nn.AddConstant(-2)
Chrom_i4=-nn.AddConstant(-1)
Chrom_i5=-nn.AddConstant(-1)
Chrom_i4r=-nn.AddConstant(-1)
Chrom_i5r=-nn.AddConstant(-1)

Chrom_i6=-nn.Identity()--Reshape(44,1)
Chrom_i7=-nn.Identity()--Reshape(44,1)
Chrom_i8=-nn.Identity()

normal_const=5e-5;

single_loci_loss=normal_const*(1-2e-1);
Chrom_h6=Chrom_i6-nn.MulConstant(-2*torch.log(single_loci_loss))
Chrom_h7=Chrom_i7-nn.MulConstant(-2*torch.log(single_loci_loss))
Chrom_t6=Chrom_i6-nn.MulConstant(-1)-nn.AddConstant(1)
Chrom_t8=Chrom_i8-nn.MulConstant(-1)-nn.AddConstant(1)


Chrom_WGD=Chrom_i1-Chrom_WGD_res
Chrom_WGD=Chrom_WGD-nn.Replicate(44,2)


Chrom_Val_noWGD=Chrom_i2-Chrom_Net_noWGD
Chrom_Val_noWGD={Chrom_Val_noWGD,Chrom_t6}-nn.CMulTable()
Chrom_Val_noWGD=Chrom_Val_noWGD-nn.Sum(2)-nn.Reshape(1)-nn.Replicate(44,2)


Chrom_Val_noWGD={Chrom_Val_noWGD,Chrom_h6}-nn.CAddTable()


Chrom_Val0_WGD=Chrom_i3-Chrom_Net_WGD

Chrom_Val1b_WGD=Chrom_i4-Chrom_Net_noWGD:clone()

--Chrom_Val2b_WGD=Chrom_i5-Chrom_Net_noWGD:clone()
--Chrom_Val1a_WGD=Chrom_i4r-Chrom_Net_noWGD:clone()
--Chrom_Val2a_WGD=Chrom_i5r-Chrom_Net_noWGD:clone()
--Chrom_Val1_WGD={Chrom_Val1b_WGD,Chrom_Val1a_WGD}-nn.CAddTable()
--Chrom_Val2_WGD={Chrom_Val2b_WGD,Chrom_Val2a_WGD}-nn.CAddTable()


--Chrom_Val_WGD={Chrom_Val0_WGD,Chrom_Val1_WGD,Chrom_Val2_WGD}-nn.JoinTable(2,2)

Softmin=nn.Sequential()
Softmin:add(nn.Replicate(2))
Softmin:add(nn.SplitTable(1))
soft=nn.ParallelTable()
soft:add(nn.Sequential():add(nn.Min(2)):add(nn.Replicate(3,2)))
soft:add(nn.Sequential():add(nn.SoftMin()))
Softmin:add(soft)
Softmin:add(nn.CMulTable())
Softmin:add(nn.Sum(2))
Softmin:add(nn.Reshape(1))

--Chrom_Val_WGD=Chrom_Val_WGD-Softmin
--Chrom_Val_WGD=Chrom_Val_WGD-nn.Replicate(44,2)
--only two
Chrom_Val_WGD={Chrom_Val0_WGD,Chrom_t8}-nn.CMulTable()
Chrom_Val_WGD=Chrom_Val_WGD-nn.Sum(2)-nn.Reshape(1)-nn.Replicate(44,2)

Chrom_Val_WGD={Chrom_Val_WGD,Chrom_h7}-nn.CAddTable()

WGD=normal_const*0.6;

Chrom_Val_WGD=Chrom_Val_WGD-nn.AddConstant(-torch.log(WGD))


Chrom_Val={Chrom_Val_noWGD,Chrom_Val_WGD}-nn.JoinTable(2,2)
Chrom_Val={Chrom_Val,Chrom_WGD}-nn.CMulTable()
Chrom_Val=Chrom_Val-nn.Sum(3)

--Chrom_Model=nn.gModule({Chrom_i1,Chrom_i2,Chrom_i3,Chrom_i4,Chrom_i4r,Chrom_i5,Chrom_i5r,Chrom_i6,Chrom_i7},{Chrom_Val})
Chrom_Model=nn.gModule({Chrom_i1,Chrom_i2,Chrom_i3,Chrom_i6,Chrom_i7,Chrom_i8},{Chrom_Val})




CNV_i1=-nn.Identity()
CNV_i1_r=-nn.MulConstant(1)
CNV_i2_r=-nn.MulConstant(1)
CNV_i3=-nn.Identity()

CNV_Net=nn.Sequential()
CNV_Net:add(nn.SpatialConvolution(1, nkernels[1]/2, 1, 7, 1, 1, 0,3))
CNV_Net:add(nn.Threshold(0, 1e-6))


CNV_Net:add(nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 7, 1, 1, 0,3))
CNV_Net:add(nn.Threshold(0, 1e-6))
CNV_Net:add(nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 7, 1, 1, 0,3))
CNV_Net:add(nn.Threshold(0, 1e-6))
CNV_Net:add(nn.SpatialConvolution(nkernels[3]/2, nkernels[4]/2, 1, 7, 1, 1, 0,3))
CNV_Net:add(nn.Threshold(0, 1e-6))
CNV_Net:add(nn.Reshape(nkernels[4]/2*50))
CNV_Net:add(nn.Linear(nkernels[4]/2*50,2*chrom_width-1))

CNV_Net_WGD=CNV_Net:clone()

CNV_Val_noWGD=CNV_i1-CNV_Net
CNV_Val_noWGD={CNV_Val_noWGD,CNV_i1_r}-nn.CAddTable()
CNV_Val_noWGD=CNV_Val_noWGD-nn.Reshape(99,1)

CNV_Val_WGD=CNV_i1-CNV_Net_WGD
CNV_Val_WGD={CNV_Val_WGD,CNV_i2_r}-nn.CAddTable()
CNV_Val_WGD=CNV_Val_WGD-nn.Reshape(99,1)


CNV_WGD=CNV_i3-Chrom_WGD_res:clone()
CNV_WGD=CNV_WGD-nn.Replicate(2*chrom_width-1,2)

CNV_Val={CNV_Val_noWGD,CNV_Val_WGD}-nn.JoinTable(2,2)
CNV_Val={CNV_Val,CNV_WGD}-nn.CMulTable()
CNV_Val=CNV_Val-nn.Sum(3)

CNV_Model=nn.gModule({CNV_i1,CNV_i1_r,CNV_i2_r,CNV_i3},{CNV_Val})











--CNV_Model:forward({torch.ones(2,1100,1),torch.ones(1,50,1)})
--determin the end position
End_Point_i1=-nn.Identity()
End_Point_i2=-nn.Identity()
End_Point_h1={End_Point_i1,End_Point_i2}-nn.JoinTable(1,3)
--End_Point_h1=End_Point_h1-nn.SpatialConvolution(1*nfeats, nkernels[1]/2, 1, 7, 1, 1, 0,3)
--End_Point_h1 =End_Point_h1-nn.Threshold(0, 1e-6)
--End_Point_Net =End_Point_Net-nn.SpatialMaxPooling(1,4,1,4,0,4-width%4)

End_Point_i3=-nn.Identity()
--End_Point_h2 =End_Point_i3-nn.SpatialConvolution(nfeats,nkernels[1]/2, 1, 5, 1, 7, 0,3)
--End_Point_h2 =End_Point_h2-nn.Threshold(0, 1e-6)
--End_Point_h2 =End_Point_h2-nn.SpatialMaxPooling(1,9,1,3,0,0)
End_Point_Net=nn.Sequential()
End_Point_Net:add(nn.SpatialConvolution(2, nkernels[1]/2, 1, 7, 1, 1, 0,3))
End_Point_Net:add(nn.Threshold(0, 1e-6))


End_Point_Net:add(nn.SpatialConvolution(nkernels[1]/2, nkernels[2]/2, 1, 7, 1, 1, 0,3))
End_Point_Net:add(nn.Threshold(0, 1e-6))
--CNV_Net =CNV_Net-nn.SpatialMaxPooling(1,5,1,5,0,5-math.ceil(width/20)%5)
End_Point_Net:add(nn.SpatialConvolution(nkernels[2]/2, nkernels[3]/2, 1, 7, 1, 1, 0,3))
End_Point_Net:add(nn.Threshold(0, 1e-6))

End_Point_Net:add(nn.Reshape(nkernels[3]/2*chrom_width))
End_Point_Net:add(nn.Linear(nkernels[3]/2*chrom_width,chrom_width-1))

End_Point_WGD=End_Point_Net:clone()

End_Point_Val_noWGD=End_Point_h1-End_Point_Net-nn.Reshape(chrom_width-1,1)
End_Point_Val_WGD=End_Point_h1-End_Point_WGD-nn.Reshape(chrom_width-1,1)

End_WGD=End_Point_i3-Chrom_WGD_res:clone()
End_WGD=End_WGD-nn.Replicate(chrom_width-1,2)

End_Point_Val={End_Point_Val_noWGD,End_Point_Val_WGD}-nn.JoinTable(2,2)
End_Point_Val={End_Point_Val,End_WGD}-nn.CMulTable()
End_Point_Val=End_Point_Val-nn.Sum(3)


--End_Point_Net= End_Point_Net-nn.SoftMax()
--End_Point_Net=End_Point_Net-nn.Sigmoid()
End_Point_Model=nn.gModule({End_Point_i1,End_Point_i2,End_Point_i3},{End_Point_Val});
