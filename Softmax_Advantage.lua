require "torch"
require "nn"
require "math"

normal_const=5e-5;
single_loci_loss=normal_const*(1-2e-1);
Half_Chromosome_CNV=normal_const*(1-0.4);
Whole_Chromosome_CNV=normal_const*(1-0.5);
WGD=normal_const*0.6;
const1=normal_const*(1-1e-1);
const2=5;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	if ChrA==1 then
		reward=torch.sum(torch.abs(StartS-2*EndS))*math.log(single_loci_loss);
		reward=reward+math.log(WGD);
	else
		reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
		--if torch.abs(EndL-StartL+1-chrom_width/2)<2 then
		--	reward=math.log(Half_Chromosome_CNV);
		--else 
			if torch.abs(EndL-StartL+1-chrom_width)<1 then
				reward=math.log(Whole_Chromosome_CNV);
			end
		--end
	end
	reward=reward+torch.sum(torch.abs(EndS-1))*math.log(single_loci_loss);
	reward=reward-torch.sum(torch.abs(StartS-1))*math.log(single_loci_loss);
	return reward;
end

Advantage_cal=function()
	Chrom_Model:forward(train.next)
	train.Advantage=train.Reward
	--train.end_loss=(((torch.abs(train.next-1)):sum(2)):sum(3)*math.log(single_loci_loss)):resize(train.Advantage:size());
	--train.Advantage=train.Advantage+Chrom_Model.output:max(2)
	--train.Advantage=train.Advantage+torch.log(torch.exp(train.end_loss-Chrom_Model.output:max(2))+torch.exp(Chrom_Model.output-(Chrom_Model.output:max(2)):expand(Chrom_Model.output:size())):sum(2))
	--force WGD
	Chrom_Model.output:select(2,1):add(torch.log(train.WGD_flag))
	train.Advantage=train.Advantage+Chrom_Model.output:max(2)
	train.Advantage=train.Advantage+torch.log(torch.exp(0-Chrom_Model.output:max(2))+torch.exp(Chrom_Model.output-(Chrom_Model.output:max(2)):expand(Chrom_Model.output:size())):sum(2))
	
	Chrom_Model:forward(train.state)	
	
	CNV_Model:forward({train.state,train.chrom_state})
	End_Point_Model:forward({train.chrom_state,train.chrom_state_new})
	for i = 1,train.state:size(1) do
		train.Advantage[i]=train.Advantage[i]-(Chrom_Model.output[i][train.ChrA[i]])
		if train.ChrA[i]>1 then
			train.Advantage[i]=train.Advantage[i]-(CNV_Model.output[i][train.CNV[i]])
			temp_sum=0
			for j=1,train.start_loci[i]:size(1) do
				temp_sum=temp_sum+torch.exp(CNV_Model.output[i][2*train.start_loci[i][j][1]+train.start_loci[i][j][2]*4-4])
				temp_sum=temp_sum+torch.exp(CNV_Model.output[i][2*train.start_loci[i][j][1]+train.start_loci[i][j][2]*4-5])
			end

			train.Advantage[i]=train.Advantage[i]+torch.log(temp_sum)

			train.Advantage[i]=train.Advantage[i]-(End_Point_Model.output[i][train.End[i]])
		--	train.Advantage[i]=train.Advantage[i]+torch.log(torch.sum(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}]))
			temp_sum=0
			for j=1,train.end_loci[i]:size(1) do
                        	temp_sum=temp_sum+torch.exp(End_Point_Model.output[i][train.end_loci[i][j][1]])
			end
			train.Advantage[i]=train.Advantage[i]+torch.log(temp_sum)
		end
	end
	return train.Advantage;
end
