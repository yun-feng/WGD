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
   learningRate=5e-5,
   learningRateDecay=1e-7,
   weightDecay=1e-8,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-10
}

opt.State_CNV=deepcopy(opt.State_Chrom)

opt.State_CNV.learningRate=1e-3
opt.State_End=deepcopy(opt.State_CNV)
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
		grad[i][train.ChrA[i]]=-train.Advantage[i]/(train.Advantage:size(1))
		 grad[i][1]=train.Advantage[i]/(train.Advantage:size(1))
	end
	
    Chrom_Model:backward(train.state,grad);

    local par_cp=parGrad_Chrom:clone();
    
    Chrom_Model:forward(train.next)

    Chrom_Model:zeroGradParameters();
    local grad=torch.zeros(Chrom_Model.output:size())
    for i= 1,grad:size(1) do
        grad[i][1]=-train.Advantage[i]/(train.Advantage:size(1))
        local temp,temp_l=(Chrom_Model.output[i]):max(1)
        grad[i][temp_l[1]]=grad[i][temp_l[1]]+train.Advantage[i]/train.Advantage:size(1)
        
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
        if train.ChrA[i]>2 then
		grad[i][train.CNV[i]]=-train.Advantage[i]/(train.Advantage:size(1))
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
        	if train.ChrA[i]>2 then
        	    grad[i][train.End[i]]=-train.Advantage[i]/(train.Advantage:size(1))
        	    local temp,temp_l=(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}]):max(1)
        	    grad[i][temp_l[1]]=grad[i][temp_l[1]]+train.Advantage[i]/(train.Advantage:size(1))
		end
    end
	
    End_Point_Model:backward({train.chrom_state,train.chrom_state_new},grad);
    
    return f,parGrad_End;
end

function model_train()
    
	local temp,losses=opt.Method(feval_Chrom,par_Chrom,opt.State_Chrom);
	local temp,losses=opt.Method(feval_CNV,par_CNV,opt.State_CNV);
	local temp,losses=opt.Method(feval_End,par_End,opt.State_End);
	
end
