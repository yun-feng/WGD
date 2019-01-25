require "torch"
require "nn"
require "math"

normal_const=5e-6;
single_loci_loss=(1-2e-1);
Half_Chromosome_CNV=(1-0.4);
Whole_Chromosome_CNV=(1-0.5);
WGD=0.6;
const1=(1-1e-1);
const2=5;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	if ChrA==2 then
		reward=torch.sum(torch.abs(StartS-2*EndS))*math.log(single_loci_loss);
		reward=reward+math.log(WGD);
	else
		reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
		if torch.abs(EndL-StartL+1-chrom_width/2)<2 then
			reward=math.log(Half_Chromosome_CNV);
		else 
			if torch.abs(EndL-StartL+1-chrom_width)<3 then
				reward=math.log(Whole_Chromosome_CNV);
			end
		end
	end
	reward=reward+torch.sum(torch.abs(EndS-1))*math.log(single_loci_loss);
	reward=reward-torch.sum(torch.abs(StartS-1))*math.log(single_loci_loss);
	return reward;
end

Advantage_cal=function()
	train.Advantage=train.Reward+math.log(normal_const);
	Chrom_Model:forward(train.next)
	train.Advantage=train.Advantage-torch.log(Chrom_Model.output:select(2,1))
	Chrom_Model:forward(train.state)	
	train.Advantage=train.Advantage+torch.log(Chrom_Model.output:select(2,1))

	CNV_Model:forward({train.state,train.chrom_state})
	End_Point_Model:forward({train.chrom_state,train.chrom_state_new})
	for i = 1,train.state:size(1) do
		train.Advantage[i]=train.Advantage[i]-torch.log(Chrom_Model.output[i][train.ChrA[i]])
		if train.ChrA[i]>2 then
			train.Advantage[i]=train.Advantage[i]-torch.log(CNV_Model.output[i][train.CNV[i]])
			train.Advantage[i]=train.Advantage[i]-torch.log(End_Point_Model.output[i][train.End[i]])
			train.Advantage[i]=train.Advantage[i]+torch.log(torch.sum(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}]))
		end
	end
	return train.Advantage;
end
