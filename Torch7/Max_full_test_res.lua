require "torch"
require "nn"
require "nngraph"

wkdir="/data/suzaku/ted/WGD/Max_"
opt={}
train={}
test={}

dofile (wkdir.."Policy_res.lua");
dofile (wkdir.."read_data_Kuramoch.lua");

--Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res3");
Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res_new3");
CNV_Model=torch.load(wkdir.."Model_CNV_Model_res_new3");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_res_new3");

dofile (wkdir.."Advantage_res.lua");
dofile (wkdir.."train_res.lua");
dofile (wkdir.."data_simu_res_loss.lua");

train.step_sample=100
Nsample=15
train.Chr_sample=torch.zeros(Nsample,train.step_sample)
train.CNV_sample=torch.zeros(Nsample,train.step_sample)
train.End_sample=torch.zeros(Nsample,train.step_sample)
train.WGD_sample=torch.zeros(Nsample)

LoadData_t(train.step_sample)

test.ChrA=torch.zeros(train.step_sample*2)
test.CNV=torch.zeros(train.step_sample*2)
test.End=torch.zeros(train.step_sample*2)
'
test.Naive=torch.zeros(380)
test.Naive2=torch.zeros(380)
for k=0,37 do
	readdata(k)
	Nsample=10
	for i=1,Nsample do
		for chr=1,44 do
			temp_allele=torch.floor((chr-1)/22)
			temp_break=1
			for j=1,50 do
				if(not (train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]==temp_break) )then
				temp_break=train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]
				test.Naive[k*10+i]=test.Naive[k*10+i]+torch.abs(temp_break-1)
				end
			end
		end
	--	test.Naive[k*10+i]=math.min(test.Naive[k*10+i],train.step_sample+20)
	end
	for i=1,Nsample do
		if (torch.sum(train.state[i])/2200>1.4) then
			test.Naive2[k*10+i]=test.Naive2[k*10+i]+1
			for chr=1,44 do
				temp_allele=torch.floor((chr-1)/22)
				temp_chr=train.state[i][{temp_allele+1,{(chr-1-temp_allele*22)*50+1,(chr-1-temp_allele*22)*50+50},1}]:clone()
				temp_chr=temp_chr-torch.floor(temp_chr/2)*2
				temp_break=0
				for j=1,50 do
					if(not (temp_chr[j]==temp_break) )then
					temp_break=temp_chr[j]
					test.Naive2[k*10+i]=test.Naive2[k*10+i]+torch.abs(temp_break)
					end
				end
			end
			train.state[i]=torch.floor(train.state[i]/2)
		end
		for chr=1,44 do
			temp_allele=torch.floor((chr-1)/22)	
			temp_ave=torch.floor(torch.sum(train.state[i][{temp_allele+1,{(chr-1-temp_allele*22)*50+1,(chr-1-temp_allele*22)*50+50},1}])/50+0.5)
			temp_break=temp_ave
			for j=1,50 do
				if(not (train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]==temp_break) )then
				temp_break=train.state[i][{temp_allele+1,(chr-1-temp_allele*22)*50+j,1}]
				test.Naive2[k*10+i]=test.Naive2[k*10+i]+torch.abs(temp_break-temp_ave)
				end
			end
			test.Naive2[k*10+i]=test.Naive2[k*10+i]+torch.abs(temp_ave-1)
		end
	--	test.Naive2[k*10+i]=math.min(test.Naive2[k*10+i],train.step_sample+20)
	end
end
			


out_file1=wkdir.."COAD_Chr_res"
out_file2=wkdir.."COAD_CNV_res"
out_file3=wkdir.."COAD_End_res"
out1 = assert(io.open(out_file1, "w")) 
out2 = assert(io.open(out_file2, "w")) 
out3 = assert(io.open(out_file3, "w")) 
splitter = "\t"
for i=0,38 do
	readdata(i)
	for j=1,10 do
		Deconvolute(train.state[j],70)
		for k=1,70 do
			out1:write(test.ChrA[k])
			out2:write(test.CNV[k])
			out3:write(test.End[k])
			out1:write(splitter)
	                out2:write(splitter)
        	        out3:write(splitter)

		end
		out1:write("\n")
		out2:write("\n")
		out3:write("\n")
	end
end
'
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
		train.state[i]=torch.floor(train.state[i]/2)
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




