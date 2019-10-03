require "torch"
require "nn"
require "math"

normal_const=1--5e-5;
single_loci_loss=normal_const*(1-2e-1);
Half_Chromosome_CNV=normal_const*(1-0.4);
Whole_Chromosome_CNV=normal_const*(1-0.5);
WGD=normal_const*0.6;
const1=normal_const*(1-1e-1);
const2=5;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	--if ChrA==1 then
	--	reward=torch.sum(torch.abs(StartS-2*EndS))*math.log(single_loci_loss);
	--	reward=reward+math.log(WGD);
	--else
		reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
		--if torch.abs(EndL-StartL+1-chrom_width/2)<2 then
		--	reward=math.log(Half_Chromosome_CNV);
		--else 
			if torch.abs(EndL-StartL+1-chrom_width)<1 then
				reward=math.log(Whole_Chromosome_CNV);
			end
		--end
	--end
	--reward=reward+torch.sum(torch.abs(EndS-1))*math.log(single_loci_loss);
	--reward=reward-torch.sum(torch.abs(StartS-1))*math.log(single_loci_loss);
	return reward;
end

WGD_LOSS=function(cnp,time)
	local reward_next;
	reward_next=-Chrom_Model:forward(cnp):min()
	if(reward_next< torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)) then
		reward_next=torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)
	end
	
	if (torch.sum(torch.abs(cnp-2*torch.floor(cnp/2))) <1 and torch.sum(cnp)>0) then
		local temp=WGD_LOSS(torch.floor(cnp/2),time)
		if (temp> reward_next) then
			reward_next=temp
			time=time+1
		end
	end
	return reward_next+math.log(WGD),time;
end	

Advantage_cal=function()
	Chrom_Model:forward(train.next)
	train.Advantage=train.Advantage+train.Reward:clone()
	--train.end_loss=(((torch.abs(train.next-1)):sum(2)):sum(3)*math.log(single_loci_loss)):resize(train.Advantage:size());
	--train.Advantage=train.Advantage+Chrom_Model.output:max(2)
	--train.Advantage=train.Advantage+torch.log(torch.exp(train.end_loss-Chrom_Model.output:max(2))+torch.exp(Chrom_Model.output-(Chrom_Model.output:max(2)):expand(Chrom_Model.output:size())):sum(2))
	--force WGD
	--Chrom_Model.output:select(2,1):add(torch.log(train.WGD_flag))
	train.max_next=-Chrom_Model.output:min(2):resize(train.Reward:size())
	train.wgd_times=torch.zeros(train.next:size(1))
	for i=1,train.state:size(1) do
			if(torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss) > train.max_next[i]) then
				train.max_next[i]=torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss)
			end
			if(train.WGD_flag[i]==1) then
				local wgd_loss,wgd_time=WGD_LOSS(train.next[i],0)
				if(wgd_loss > train.max_next[i]) then
					train.max_next[i]=wgd_loss
					train.wgd_times[i]=wgd_time
					while(wgd_time>0) do
						train.next=torch.floor(train.next/2)
						wgd_time=wgd_time-1
					end
				end
			end
	end
				
	train.Advantage=train.Advantage+torch.cmul(train.max_next,train.valid);
	
	Chrom_Model:forward(train.state)	
	
	CNV_Model:forward({train.state,train.chrom_state})
	End_Point_Model:forward({train.chrom_state,train.chrom_state_new})
	for i = 1,train.state:size(1) do

if train.valid[i]>0 then
		train.Advantage[i]=train.Advantage[i]+(Chrom_Model.output[i][train.ChrA[i]])
		--if train.ChrA[i]>1 then
			if train.CNV[i]>1 then
				train.Advantage[i]=train.Advantage[i]-2*(CNV_Model.output[i][train.CNV[i]-1])
			end
			temp_max=CNV_Model.output[i][train.start_loci[i][1][2]*2-1]
			for j=1,train.start_loci[i]:size(1) do
				temp_max=math.max(temp_max,CNV_Model.output[i][train.start_loci[i][j][2]*2-1])
				if (train.chrom_state[i][1][train.start_loci[i][j][2]][1]-1>0) then
					if  train.start_loci[i][j][2]*2-1>1 then
				 		temp_max=math.max(temp_max,CNV_Model.output[i][train.start_loci[i][j][2]*2-1-1])
					else
						temp_max=math.max(temp_max,0)
					end
				end
			end

			train.Advantage[i]=train.Advantage[i]+2*temp_max
			if train.End[i]>1 then
				train.Advantage[i]=train.Advantage[i]-2*(End_Point_Model.output[i][train.End[i]-1])
			end
		--	train.Advantage[i]=train.Advantage[i]+torch.log(torch.sum(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}]))
			if train.end_loci[i][1][1]>1 then
				temp_max=End_Point_Model.output[i][train.end_loci[i][1][1]-1]
			else
				temp_max=0
			end
			for j=1,train.end_loci[i]:size(1) do
				if(train.cnv[i]>0 or train.chrom_state[i][1][train.end_loci[i][j][1]][1]-1>0) then
					if train.end_loci[i][j][1] >1 then
                        			temp_max=math.max(temp_max,(End_Point_Model.output[i][train.end_loci[i][j][1]-1]))
					else
						temp_max=math.max(temp_max,0)
					end
				end
			end
			train.Advantage[i]=train.Advantage[i]+2*temp_max
end
	end
	return train.Advantage;
end
