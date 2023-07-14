 #!/bin/bash

# phonological and visual world inputs
trfname='W200_I2_01_jaccard_selected'
mapping='phnvwd2sem'
nvwd=2
distractors='none'

# settings for the features
ptimeincs=2
vsplits=5
vtimeincs=4
segchars="|"
padding="half"
tgtfull=1 # 1: set target on for the entire duration of the word unfolding
# settings for the training
gpuindex="0"
startepoch=0
precision="single"
nepochs=500000
necrit=50000 
modelstart=1
nbatches=1
batchlen=1

# settings for input hidden layer
phnihiddim=0 
phnihidnlayers=0 
phnihidndirs=0 
# settings for output layer
hubnlayers=1
hubndirs=0

outputfile="outputs/output"
errorfile="errors/error"

batchstart=${modelstart}
for i in $(seq 1 ${nbatches}); do
  echo "Starting batch "${i}
  batchstop=$[ batchstart + batchlen - 1]
  for imdl in $(seq ${batchstart} ${batchstop}); do
    echo 'Starting model '${imdl}
    python3 train_wrapper.py ${nvwd} ${trfname} ${distractors} ${mapping} \
                             ${ptimeincs} ${vsplits} ${vtimeincs} ${segchars} \
                             ${imdl} ${imdl} ${startepoch} ${nepochs} ${necrit} \
                             ${padding} \
                             ${precision} ${gpuindex} \
                             ${phnihiddim} ${phnihidnlayers} ${phnihidndirs} ${hubnlayers} ${hubndirs} \
                             ${tgtfull} 1> "${outputfile}_${imdl}.txt" 2> "${errorfile}_${imdl}.txt" &
  done
  batchstart=$[ batchstart + batchlen ]
  wait
done
