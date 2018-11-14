require "torch"
require "nn"
require "math"

single_loci_loss=1-1e-4;
Half_Chromosome_CNV=1-0.6;
Whole_Chromosome_CNV=1-0.3;
WGD=0.6;
const1=1;
const2=0.1;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	if ChrA==1 then
		reward=torch.sum(torch.abs(StartS-1))*math.log(single_loci_loss);
	else 
		if chrA==2 then
			reward=torch.sum(torch.abs(StartS-2*EndS))*math.log(single_loci_loss);
			reward=reward+math.log(WGD);
			reward=reward+ValueNet_eval:forward(EndS);
		else
			reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
			if torch.abs(EndL-StartL+1-chrom_width/2)<2 then
				reward=math.log(Half_Chromosome_CNV);
			else 
				if torch.abs(EndL-StartL+1-chrom_width)<3 then
					reward=math.log(Whole_Chromosome_CNV);
				end
			end
			reward=reward+(ValueNet_eval:forward(EndS))[1];
		end
	end

	return reward;
end

Advantage_cal=function()
	ValueNet:forward(train.state);
	train.Advantage=train.Reward-ValueNet.output:select(2,1);
	
	return train.Advantage;
end
