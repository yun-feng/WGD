require "torch"
require "nn"
require "nngraph"

wkdir="/data/suzaku/ted/WGD/Max_"
opt={}
train={}
test={}

dofile (wkdir.."Policy_res2.lua");
dofile(wkdir.."read_data_Kuramoch.lua");

Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res");
CNV_Model=torch.load(wkdir.."Model_CNV_Model_res");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_res");

dofile (wkdir.."Advantage_res2.lua");
dofile (wkdir.."train_res2.lua");
dofile (wkdir.."data_simu_res2.lua");

train.step_sample=100
Nsample=10
train.Chr_sample=torch.zeros(Nsample,train.step_sample)
train.CNV_sample=torch.zeros(Nsample,train.step_sample)
train.End_sample=torch.zeros(Nsample,train.step_sample)
train.WGD_sample=torch.zeros(Nsample)

LoadData_t(train.step_sample)

test.ChrA=torch.zeros(train.step_sample*2)
test.CNV=torch.zeros(train.step_sample*2)
test.End=torch.zeros(train.step_sample*2)



--Deconvolute(train.state[2],train.step_sample*2)

--torch.sum(torch.abs(train.state[2]-1))

Nsample=10
test.Naive=torch.zeros(Nsample)
for i=1,Nsample do
	for chr=1,44 do
		temp_allele=torch.floor((chr-1)/22)
		temp_break=1
		for j=1,50 do
			if(not (train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]==temp_break) )then
			temp_break=train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]
			test.Naive[i]=test.Naive[i]+torch.abs(temp_break-1)
			end
		end
	end
	test.Naive[i]=math.min(test.Naive[i],train.step_sample+20)
end

test.Naive2=torch.zeros(Nsample)
temp_state=train.state:clone()
for i=1,Nsample do
	if (torch.sum(train.state[i])/2200>1.7) then
		test.Naive2[j*10+i]=test.Naive2[j*10+i]+1
		for chr=1,44 do
			temp_allele=torch.floor((chr-1)/22)
			temp_chr=train.state[i][{temp_allele+1,{(chr-1-temp_allele*22)*50+1,(chr-1-temp_allele*22)*50+50},1}]:clone()
			temp_chr=temp_chr-torch.floor(temp_chr/2)*2
			temp_break=0
			for j=1,50 do
				if(not (temp_chr[j]==temp_break) )then
				temp_break=temp[j]
				test.Naive2[i]=test.Naive2[i]+torch.abs(temp_break)
				end
			end
		end
		train.state[i]=torch.floor(train.state[i]/2)*2
	end
	for chr=1,44 do
		temp_allele=torch.floor((chr-1)/22)	
		temp_ave=torch.floor(torch.sum(train.state[i][{temp_allele+1,{(chr-1-temp_allele*22)*50+1,(chr-1-temp_allele*22)*50+50},1}])/50+0.5)
		temp_break=temp_ave
		for j=1,50 do
			if(not (train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]==temp_break) )then
			temp_break=train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]
			test.Naive2[i]=test.Naive2[i]+torch.abs(temp_break-temp_ave)
			end
		end
		test.Naive2[i]=test.Naive2[i]+torch.abs(temp_ave-1)
	end
	test.Naive2[i]=math.min(test.Naive2[i],train.step_sample+20)
end

train.state=temp_state

test.truth=torch.zeros(Nsample)
test.rl=torch.zeros(Nsample)
for i=1,Nsample do
	cnp_a=train.state[i]:clone()
	test.ChrA=torch.zeros(train.step_sample+20)
	test.CNV=torch.zeros(train.step_sample+20)
	test.End=torch.zeros(train.step_sample+20)
	Deconvolute(train.state[i],train.step_sample+20)
	test.rl[i]=test.rl[i]+0.0+test.ChrA:nonzero():size(1)
	test.truth[i]=math.min(test.rl[i],test.Naive[i],train.Chr_sample[i]:nonzero():size(1)+torch.ceil(train.WGD_sample[i]/train.step_sample))
	if(test.rl[i]==test.truth[i]) then
	--	break
	end
end




