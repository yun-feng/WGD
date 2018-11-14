require "torch"
require "nn"

single_loci_loss=1-1e-4;
Half_Chromosome_CNV=1-0.6;
Whole_Chromosome_CNV=1-0.3;
WGD=0.6;
const1=1;
const2=0.1;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	if ChrA==1 then
		reward=torch.sum(torch.abs(StartS-1))*log(single_loci_loss);
	else 
		if chrA==2 then
			reward=torch.sum(torch.abs(StartS-2*EndS))*log(single_loci_loss);
			reward=reward+log(WGD);
			reward=reward+ValueNet_eval:forward(EndS);
		else:
			reward=log(const1/(const2+log(EndL-StartL)));
			if torch.abs(EndL-StartL-chrom_width/2)<2 then
				reward=log(Half_Chromosome_CNV);
			else 
				if torch.abs(EndL-StartL-chrom_width)<3 then
					reward=log(Whole_Chromosome_CNV);
				end
			end
			reward=reward+ValueNet_eval:forward(EndS);
		end
	end

	return reward;
end

Advantage_cal=function()
	ValueNet:forward(train.state);
	train.Advantage=train.Reward-ValueNet.output:select(2,1);
	
	return train.Advantage;
end
