#simulate data for training WGD model
setwd("/data/ted/WGD")

nsample=500;

wkdata<-matrix(rep(1,2*nsample*22*50),ncol=50*22,nrow=nsample*2)

change_p=c(1,1,-1,0,0,0,0,0,0)
change_m=c(0,0,0,-1,-1,1,0,0,0)

nstep1=10
nstep2=10
#before
for(j in 1:nstep1){
  for( i in 1:nsample){
    start=floor(runif(1,1,22*50+1))
    end=floor(runif(1,1,22*50+1))
    if(start>end){
      temp=end
      end=start
      start=temp
    }
    type=floor(runif(1,1,10))
    wkdata[i*2-1,start:end]=wkdata[i*2-1,start:end]+change_m[type]
    wkdata[i*2,start:end]=wkdata[i*2,start:end]+change_p[type]
  }
  wkdata[which(wkdata<0)]=0
}
#WGD
for( i in 1:nsample){
  if(runif(1)>0.6){
    wkdata[i*2-1,]=wkdata[i*2-1,]*2
    wkdata[i*2,]=wkdata[i*2,]*2
  }
}
#after
for(j in 1:nstep2){
  for( i in 1:nsample){
    start=floor(runif(1,1,22*50+1))
    end=floor(runif(1,1,22*50+1))
    if(start>end){
      temp=end
      end=start
      start=temp
    }
    type=floor(runif(1,1,10))
    wkdata[i*2-1,start:end]=wkdata[i*2-1,start:end]+change_m[type]
    wkdata[i*2,start:end]=wkdata[i*2,start:end]+change_p[type]
  }
  wkdata[which(wkdata<0)]=0
}

write.table(wkdata,"Simulation_cnp.txt",sep="\t",quote=F,row.names=F,col.names=F)
wkdata<-read.table("Simulation_cnp.txt",sep="\t",header=F)
