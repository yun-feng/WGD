require "torch"

require "math"

require "optim"


opt.KernelMax=0.9

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

opt.State_Chrom = {
   learningRate=1e-3,
   learningRateDecay=1e-7,
   weightDecay=1e-8,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}

opt.State_CNV=deepcopy(opt.State_Chrom)

--opt.State_CNV.learningRate=5e-5
opt.State_End=deepcopy(opt.State_CNV)
--opt.State_End.learningRate=2e-6
--opt.State_Val.learningRate=0.001

opt.Method = optim.adam;




par_Chrom,parGrad_Chrom=Chrom_Model:getParameters();

feval_Chrom=function(x)
    if x~=par_Chrom then
        par_Chrom:copy(x)
    end
    
    
    Chrom_Model:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#Chrom_Model.modules do
        if string.find(tostring(Chrom_Model.modules[i]), 'SpatialConvolution') then
                Chrom_Model.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
	
    local f=train.Advantage;
    
	local grad=torch.zeros(Chrom_Model.output:size())
	for i= 1,grad:size(1) do
		grad[i][train.ChrA[i]]=train.Advantage[i]/(train.Advantage:size(1))
	end
	
    Chrom_Model:backward(train.state,grad);

    local par_cp=parGrad_Chrom:clone();
    
    Chrom_Model:zeroGradParameters();
    Chrom_Model:forward(train.next)
   -- Chrom_Model.output:select(2,1):add(torch.log(train.WGD_flag))
    local grad=torch.zeros(Chrom_Model.output:size())
    for i= 1,grad:size(1) do
        local temp,temp_l=Chrom_Model.output[i]:min(1)--+torch.log(torch.exp(0-Chrom_Model.output[i]:max())+torch.exp(Chrom_Model.output[i]-Chrom_Model.output[i]:max()):sum())
              	if( -temp[1] > train.max_next[i]-train.wgd_times[i]*math.log(WGD)-1e-5 ) then
			grad[i][temp_l[1]]=grad[i][temp_l[1]]-train.Advantage[i]/train.Advantage:size(1)
		end
    end

    Chrom_Model:backward(train.next,grad);
    parGrad_Chrom:add(par_cp);

    return f,parGrad_Chrom;
end

par_CNV,parGrad_CNV=CNV_Model:getParameters()

feval_CNV=function(x)
    if x~=par_CNV then
        par_CNV:copy(x)
    end
    
    
    CNV_Model:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#CNV_Model.modules do
        if string.find(tostring(CNV_Model.modules[i]), 'SpatialConvolution') then
                CNV_Model.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
	
    local f=train.Advantage;
	local grad=torch.zeros(CNV_Model.output:size())
	for i= 1,grad:size(1) do
        if train.ChrA[i]>1 then
		grad[i][train.CNV[i]]=-train.Advantage[i]/(train.Advantage:size(1))
		
		local temp=CNV_Model.output[i][train.start_loci[i][1][1]*2+train.start_loci[i][1][2]*4-4]
		local temp_l=train.start_loci[i][1][1]*2+train.start_loci[i][1][2]*4-4
                for j=1,train.start_loci[i]:size(1) do
                        if((CNV_Model.output[i][train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-4]) > temp) then
				temp_l=train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-4
				temp=(CNV_Model.output[i][train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-4])
			elseif ((CNV_Model.output[i][train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-5]) > temp) then
                                temp_l=train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-5
                                temp=(CNV_Model.output[i][train.start_loci[i][j][1]*2+train.start_loci[i][j][2]*4-5])
			end
			grad[i][temp_l]=grad[i][temp_l]+train.Advantage[i]/(train.Advantage:size(1))
			
                end

	end
    end
    CNV_Model:backward({train.state,train.chrom_state},grad);
    return f,parGrad_CNV;
end

par_End,parGrad_End=End_Point_Model:getParameters()

feval_End=function(x)
    if x~=par_End then
        par_End:copy(x)
    end
    
    
    End_Point_Model:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#End_Point_Model.modules do
        if string.find(tostring(End_Point_Model.modules[i]), 'SpatialConvolution') then
                End_Point_Model.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
	
    local f=train.Advantage;
    
	local grad=torch.zeros(End_Point_Model.output:size())
	for i= 1,grad:size(1) do
        	if train.ChrA[i]>1 then
        	    grad[i][train.End[i]]=-train.Advantage[i]/(train.Advantage:size(1))
        	    --local temp=torch.sum(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}])
        	local temp=End_Point_Model.output[i][train.end_loci[i][1][1]]
		local temp_l=train.end_loci[i][1][1]
		for j=1,train.end_loci[i]:size(1) do
			if( temp<End_Point_Model.output[i][train.end_loci[i][j][1]]) then
				temp_l=train.end_loci[i][j][1]
				temp=End_Point_Model.output[i][train.end_loci[i][j][1]]
                	end
		end
                    grad[i][temp_l]=grad[i][temp_l]+train.Advantage[i]/(train.Advantage:size(1))
                

		--for j=train.StartL[i],chrom_width do
		
        	--        grad[i][j]=grad[i][j]+train.Advantage[i]/(temp*train.Advantage:size(1))
		--   end
		end
    end
	
    End_Point_Model:backward({train.chrom_state,train.chrom_state_new},grad);
    
    return f,parGrad_End;
end

function model_train()
--	local temp,losses=opt.Method(feval_Chrom,par_Chrom,opt.State_Chrom);
	local temp,losses=opt.Method(feval_CNV,par_CNV,opt.State_CNV);
	local temp,losses=opt.Method(feval_End,par_End,opt.State_End);
	
end
