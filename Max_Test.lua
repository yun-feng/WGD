##test the behavior of networks

require "torch"
require "nn"
require "nngraph"

wkdir="/data/ted/WGD/Max_"
opt={}
train={}

dofile (wkdir.."Policy.lua");
--dofile (wkdir.."Advantage.lua");
--dofile (wkdir.."train.lua");
--dofile (wkdir.."data.lua");

Chrom_Model=torch.load(wkdir.."Model_Chrom_Model");
CNV_Model=torch.load(wkdir.."Model_CNV_Model");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model");

dofile (wkdir.."Advantage2.lua");
dofile (wkdir.."train2.lua");
dofile (wkdir.."data_simu2.lua");
counter=100000

Chrom_Model.output

LoadData(1)

Chrom_Model:forward(train.state)

a,b=Chrom_Model.output:min(2)
x=torch.Tensor(25)
for i=1,25 do
x[i]=b[i]-train.ChrA[i]
end

test_chr=function(step)
	chr_set=torch.Tensor(25,step)
	LoadData(1)
	chr_set:select(2,1):copy(train.ChrA)
	for i =1,step-1 do
		LoadData_f(false)
		chr_set:select(2,i+1):copy(train.ChrA)
	end
	Chrom_Model:forward(train.state)
	a,b=Chrom_Model.output:min(2)
	b=torch.DoubleTensor(25):copy(b:select(2,1))
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
