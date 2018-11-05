require "torch"

require "math"

require "optim"


opt.KernelMax=0.9

opt.StateVal = {
   learningRate=0.001,
   learningRateDecay=1e-5,
   weightDecay=1e-6,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}
opt.StateChrom = {
   learningRate=0.001,
   learningRateDecay=1e-5,
   weightDecay=1e-6,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}
opt.StatePolicy = {
   learningRate=0.001,
   learningRateDecay=1e-5,
   weightDecay=1e-6,
   beta1=0.9,
   beta2=0.99,
   epsilon=1e-8
}
opt.Method = optim.adam;





ValueNet:training();

par,parGrad=ValueNet:getParameters();

old_par=par:clone()
new_par=par:clone()

criterion=nn.MSECriterion()

feval=function(x)
    if x~=par then
        par:copy(x)
    end
    
    
    ValueNet:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#ValueNet.modules do
        if string.find(tostring(ValueNet.modules[i]), 'SpatialConvolutionMM') then
                ValueNet.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
    
	old_par=(1-0.001)*old_par+0.001*new_par
	par=old_par
	targets=train.reward+ValueNet:forward(train.next)
	par=new_par
	
    local f=criterion:forward(ValueNet:forward(train.state),targets);
    
    model:backward(train.state,criterion:backward(ValueNet.output,targets));
    
    return f,parGrad;
end

ChromNet:training();
UpperPolicyNet:training();

Chrom_par,Chrome_parGrad=ChromNet:getParameters();
Policy_par,Policy_parGrad=UpperPolicyNet:getParameters();

Chrom_eval=function(x)
    if x~=Chrom_par then
        Chrom_par:copy(x)
    end
    
    
    ChromNet:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#ChromNet.modules do
        if string.find(tostring(ChromNet.modules[i]), 'SpatialConvolutionMM') then
                ChromNet.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
    
	par=old_par
	targets=train.reward+ValueNet:forward(train.next)-ValueNet:forward(train.state)
	par=new_par
	
    local f=targets;
    
	local grad=torch.Tensor(22+2)
	grad:zero()
	grad:indexFill(2,torch.LongTensor{train.ChromAct},1)
    ChromNet:backward(train.state,grad);
    
    return f,Chrom_parGrad;
end

Policy_eval=function(x)
    if x~=Policy_par then
        Policy_par:copy(x)
    end
    
    
    PolicyNet:zeroGradParameters();
    
	--normalization for kernal 
    for i = 1,#UpperPolicyNet.modules do
        if string.find(tostring(UpperPolicy.modules[i]), 'SpatialConvolutionMM') then
                UpperPolicyNet.modules[i].weight:renorm(2,1,opt.KernelMax)
        end
    end
    
	par=old_par
	targets=train.reward+ValueNet:forward(train.next)-ValueNet:forward(train.state)
	par=new_par
	
    local f=targets;
    
	local grad=torch.Tensor(5*50+2)
	grad:zero()
	grad:indexFill(2,torch.LongTensor{train.FocusAct},1)
	grad=grad[1][3:252]
    UpperPolicyNet:backward({train.state,train.focus},grad);
    
    return f,Policy_parGrad;
end

function model_train()
    
	local temp,losses=opt.Method(feval,par,opt.StateVal);
	
	opt.Method(Chrom_eval,Chrom_par,opt.StateChrom);
	
	opt.Method(Policy_eval,Policy_par,opt.StatePolicy);
    
    return losses;
end
    