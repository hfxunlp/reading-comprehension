require "nn"
require "deps.SequenceContainer"
require "deps.PScore"

tmod = nn.PScore(nn.Linear(100, 1, false))
pid = torch.randn(8, 100)
qid = torch.randn(6, 1, 50)
rs = tmod:forward({pid, qid})
print(rs:size())

gp, gq = unpack(tmod:updateGradInput({pid, qid}, rs))
tmod:accGradParameters({pid, qid}, rs, scale)
print(gp:size())
print(gq:size())
