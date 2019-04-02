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
   learningRate=1e-4,
   learningRateDecay=1e-7,
   weightDecay=1e-8,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}

opt.State_CNV=deepcopy(opt.State_Chrom)

opt.State_CNV.learningRate=1e-4
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
    Chrom_Model:forward(train.state)
	--normalization for kernal 
    for i = 1,15 do--#Chrom_Model.modules do
        if string.find(tostring(Chrom_Model.modules[i]), 'SpatialConvolution') then
                Chrom_Model.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
	
    local f=train.Advantage2;
    
	local grad=torch.zeros(Chrom_Model.output:size())
	for i= 1,grad:size(1) do
	if train.valid[i]>0 then
		grad[i][train.ChrA[i]]=train.Advantage2[i]/(train.Advantage:size(1))
		if(train.WGD[i]<0.5) then
			temp_chr=torch.Tensor(44):copy(torch.gt(torch.abs((train.state[i]-1)):resize(2,22,50):sum(3),0.5):resize(44))
		else
			temp_chr=torch.Tensor(44):copy(torch.gt(torch.abs((train.state[i]-torch.floor(train.state[i]/2)*2)):resize(2,22,50):sum(3),0.5):resize(44))
		end

	      local	temp=Chrom_Model.output[i]-(Chrom_Model.output[i][train.ChrA[i]]-train.Advantage2[i])
		temp=torch.cmul(nn.ReLU():forward(temp),temp_chr)-torch.cmul(nn.ReLU():forward(-temp),(1-temp_chr))
		grad[i]=grad[i]+temp/(train.Advantage:size(1))
	end
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
    CNV_Model:forward({train.state,train.chrom_state})
	--normalization for kernal 
    for i = 1,#CNV_Model.modules do
        if string.find(tostring(CNV_Model.modules[i]), 'SpatialConvolution') then
                CNV_Model.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
	
    local f=train.Advantage;
	local grad=torch.zeros(CNV_Model.output:size())
	for i= 1,grad:size(1) do
		if train.valid[i]>0 then
        	if train.CNV[i]>1 then
				grad[i][train.CNV[i]-1]=-2*train.Advantage[i]/(train.Advantage:size(1))
			end
		
			if train.max_cnv[i]>0 then
				grad[i][train.max_cnv[i]]=grad[i][train.max_cnv[i]]+2*train.Advantage[i]/(train.Advantage:size(1))
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
		if train.valid[i]>0 then
        	if train.End[i]>1 then
        	    grad[i][train.End[i]-1]=-2*train.Advantage[i]/(train.Advantage:size(1))
        	end
		    --local temp=torch.sum(End_Point_Model.output[{i,{train.StartL[i],chrom_width}}])
			local temp,templ	
			if train.end_loci[i][1][1]>1 then
				temp=End_Point_Model.output[i][train.end_loci[i][1][1]-1]
				temp_l=train.end_loci[i][1][1]-1
			else
				temp=0
				temp_l=0
			end
			for j=1,train.end_loci[i]:size(1) do
				if(train.end_loci[i][j][1]>1 and (train.cnv[i]>0 or train.chrom_state[i][1][train.end_loci[i][j][1]][1]-1>-0.5) and temp<End_Point_Model.output[i][train.end_loci[i][j][1]-1]) then
					temp_l=train.end_loci[i][j][1]-1
					temp=End_Point_Model.output[i][temp_l]
				end
			end
			if temp_l>0 then
				grad[i][temp_l]=grad[i][temp_l]+2*train.Advantage[i]/(train.Advantage:size(1))
			end
		end
    end
	
    End_Point_Model:backward({train.chrom_state,train.chrom_state_new,train.state},grad);
    
    return f,parGrad_End;
end

function model_train()
	local temp,losses=opt.Method(feval_Chrom,par_Chrom,opt.State_Chrom);
	local temp,losses=opt.Method(feval_CNV,par_CNV,opt.State_CNV);
	local temp,losses=opt.Method(feval_End,par_End,opt.State_End);
end
