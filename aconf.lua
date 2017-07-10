starterate=math.huge--warning:only used as init erate, not asigned to criterion

runid="170710_ne_2gaoabase_kvuv_088sgd_mod2_01_v50_h50"
loadid=nil or runid--"170710_ne_2gaoabase_kv_088sgd_mod2_01_v50_h50"
--runid="debug"
logd="logs"

ieps=1

modlr=1/128--8192 16384 32768/2

warmcycle=0--1
warmlr=nil--modlr/8
keepv=nil--1
rupdv=nil--1

expdecaycycle=8
gtraincycle=32

earlystop=gtraincycle

csave=3

lrdecaycycle=8

recyclemem=nil--0.05

storedebug=true--store model every epoch or not

cntrain=nil--"modrs/"..loadid.."/devnnmod1.asc"

modtrain=2--nil

partrain=nil--1000
partupdate=10000
partsilent=false

if cntrain then
	warmcycle=0
end
