require "nn"
require "models.QAM"

function clearModule(module)
	module:apply(function(m)
		if m.clearState and not torch.isTypeOf(m, "nn.gModule") then
			m:clearState()
		end
	end)
	return module
end

wvec = torch.randn(64, 32)
m = buildQAM(1)

m:evaluate()
require "cutorch"
require "cunn"
require "cudnn"
m:cuda()
p={torch.IntTensor({1, 2, 3}):reshape(3, 1):cudaLong(), torch.IntTensor({1, 2, 3, 4}):reshape(4, 1):cudaLong(), torch.IntTensor({1, 2, 3, 4, 5}):reshape(5, 1):cudaLong()}
q=torch.IntTensor({1, 2, 3, 4}):reshape(4, 1):cudaLong()
input={p, q}
output=m:forward(input)
m:training()
output=m:forward(input)
g=m:backward(input, output)

m=clearModule(m)

output=m:forward(input)

g=m:backward(input, output)

m=clearModule(m)

--torch.save("test.asc", m, "binary", false)
torch.save("test.asc", m, "binary", true)
