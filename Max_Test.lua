##test the behavior of networks

require "torch"
require "nn"
require "nngraph"

wkdir="/data/ted/WGD/Max_"
opt={}
train={}

dofile (wkdir.."Policy_res.lua");
batch_sample=10
--dofile (wkdir.."Advantage.lua");
--dofile (wkdir.."train.lua");
--dofile (wkdir.."data.lua");

Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res");
CNV_Model=torch.load(wkdir.."Model_CNV_Model_res");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_res");

dofile (wkdir.."Advantage_res.lua");
dofile (wkdir.."train_res.lua");
dofile (wkdir.."data_simu_res.lua");
counter=100000

Chrom_Model.output

LoadData(1)

Chrom_Model:forward(train.state_cal)

a,b=Chrom_Model.output:min(2)
x=torch.Tensor(batch_sample)
for i=1,batch_sample do
x[i]=b[i]-train.ChrA[i]
end

test_chr=function(step)
	chr_set=torch.Tensor(batch_sample,step)
	LoadData(1)
	chr_set:select(2,1):copy(train.ChrA)
	for i =1,step-1 do
        LoadData_f(false)
        chr_set:select(2,i+1):copy(train.ChrA)
	end
	Chrom_Model:forward(train.state)
	a,b=Chrom_Model.output:min(2)
	b=torch.DoubleTensor(batch_sample):copy(b:select(2,1))
	s=chr_set:select(2,1)-b
	for i =1,step-1 do
        s=torch.cmul(s,chr_set:select(2,i+1)-b)
        s=s:abs()
        s=torch.ceil(s/1000)
	end
end

train.next=torch.ones(1,2,22*50,1)
flag=false
LoadData(flag)

train.next=torch.ones(1,2,22*50,1)
for i =1,10 do
	train.next[1][1][i][1]=0
end
LoadData(flag)
