starterate=math.huge--warning:only used as init erate, not asigned to criterion

runid="170527_gru_l2cqf_aoacoll"
logd="logs"

ieps=1
warmcycle=0
expdecaycycle=4
gtraincycle=32

modlr=1/8192--1024

earlystop=gtraincycle

csave=3

lrdecaycycle=4

recyclemem=0.05

storedebug=true--store model every epoch or not

cntrain=nil--"modrs/170525_gru_l2qf_maxcoll/devnnmod3.asc"

if cntrain then
	warmcycle=0
end
