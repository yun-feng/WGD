require "torch"
require "nn"
require "math"

single_loci_loss=1-2e-1;
Half_Chromosome_CNV=1-0.4;
Whole_Chromosome_CNV=1-0.5;
WGD=0.6;
const1=1-2e-1;
const2=1;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	opt.ChrA[ChrA]=opt.ChrA[ChrA]+1
	if ChrA==1 then
		reward=torch.sum(torch.abs(StartS-1))*math.log(single_loci_loss);
		reward=reward+100*math.sqrt(math.log(counter+1)/opt.ChrA[ChrA])
	elseif ChrA==2 then
		reward=torch.sum(torch.abs(StartS-2*EndS))*math.log(single_loci_loss);
		reward=reward+math.log(WGD);
		reward=reward-ValueNet_eval:forward(EndS);
		reward=reward+100*math.sqrt(math.log(counter+1)/opt.ChrA[ChrA])
	else
		opt.StartL[StartL]=opt.StartL[StartL]+1
		opt.End[EndL]=opt.End[EndL]+1
		reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
		if torch.abs(EndL-StartL+1-chrom_width/2)<2 then
			reward=math.log(Half_Chromosome_CNV);
		else 
			if torch.abs(EndL-StartL+1-chrom_width)<3 then
				reward=math.log(Whole_Chromosome_CNV);
			end
		end
		reward=reward-(ValueNet_eval:forward(EndS))[1];
		reward=reward+100*math.sqrt(math.log(counter+1)/(opt.StartL[StartL]*opt.End[EndL])^(1/2))
	end

	return reward;
end

Advantage_cal=function()
	ValueNet:forward(train.state);
	train.Advantage=train.Reward+ValueNet.output:select(2,1);
	return train.Advantage;
end

opt.ChrA=torch.zeros(22+2)
opt.StartL=torch.zeros(50*22)
opt.End=torch.zeros(50*22)
