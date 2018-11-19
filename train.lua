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

opt.State_Val = {
   learningRate=0.001,
   learningRateDecay=1e-5,
   weightDecay=1e-6,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}

opt.State_Val_eval=deepcopy(opt.State_Val)
opt.State_Val_eval.learningRate=0.05
opt.State_Chrom=deepcopy(opt.State_Val)
opt.State_CNV=deepcopy(opt.State_Val)
opt.State_End=deepcopy(opt.State_Val)

opt.Method = optim.adam;




par_Val,parGrad_Val=ValueNet:getParameters();


feval_Val=function(x)
    if x~=par_Val then
        par_Val:copy(x)
    end
    
    
    ValueNet:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#ValueNet.modules do
        if string.find(tostring(ValueNet.modules[i]), 'SpatialConvolution') then
                ValueNet.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
    
	local f=0.5*train.Advantage:pow(2);
	ValueNet:backward(train.state,torch.Tensor(train.Advantage):resize(train.Advantage:size(1),1));
    return f,parGrad_Val;
end

par_Val_eval,parGrad_Val_eval=ValueNet_eval:getParameters();
feval_Val_eval=function(x)
    if x~=par_Val then
        par_Val_eval:copy(x)
    end
    
    
    ValueNet_eval:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#ValueNet_eval.modules do
        if string.find(tostring(ValueNet_eval.modules[i]), 'SpatialConvolution') then
                ValueNet_eval.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
    
	
	local f=0.5*torch.norm(par_Val-par_Val_eval)^2;
	parGrad_Val_eval=-par_Val+par_Val_eval;
    
    return f,parGrad_Val_eval;
end


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
		grad[i][train.ChrA[i]]=train.Advantage[i]/Chrom_Model.output[i][train.ChrA[i]]
	end
	
    Chrom_Model:backward(train.state,grad);
    
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
			grad[i][train.CNV[i]]=train.Advantage[i]/CNV_Model.output[i][train.CNV[i]]
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
			grad[i][train.End[i]]=train.Advantage[i]/End_Point_Model.output[i][train.End[i]]
		end
	end
	
    End_Point_Model:backward({train.chrom_state,train.chrom_state_new},grad);
    
    return f,parGrad_End;
end

function model_train()
    
	local temp,losses=opt.Method(feval_Val,par_Val,opt.State_Val);
	local temp,losses=opt.Method(feval_Val_eval,par_Val_eval,opt.State_Val_eval);
	local temp,losses=opt.Method(feval_Chrom,par_Chrom,opt.State_Chrom);
	local temp,losses=opt.Method(feval_CNV,par_CNV,opt.State_CNV);
	local temp,losses=opt.Method(feval_End,par_End,opt.State_End);
	
end
    
