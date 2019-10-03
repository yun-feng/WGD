    
require "torch";

Major_file="/data/ted/WGD/major.txt"
Minor_file="/data/ted/WGD/minor.txt"

Num_sam=380;

function readdata(x)
    
    local majorcp = { }
    majorfile = io.open(Major_file,"r")
    if majorfile then
        for cnp in majorfile:lines() do
                table.insert(majorcp,cnp:split("\t"));
        end
        
    end
    
	local minorcp = { }
	minorfile = io.open(Minor_file,"r")
    if minorfile then
        for cnp in minorfile:lines() do
                table.insert(minorcp,cnp:split("\t"));
        end
        
    end

		
	majorcnp=torch.Tensor(majorcp);
    majorcp=nil;
	minorcnp=torch.Tensor(minorcp);
    minorcp=nil;
    
	majorcnp=majorcnp:transpose(1,2):reshape(Num_sam,1100,1)
	minorcnp=minorcnp:transpose(1,2):reshape(Num_sam,1100,1)
   
	train.state=torch.ones(10,2,1100,1)
	
	train.state:select(2,1):copy(majorcnp[{{10*x+1,10*x+10},{1,1100},{1}}])
	train.state:select(2,2):copy(minorcnp[{{10*x+1,10*x+10},{1,1100},{1}}])
	
	
	
end


