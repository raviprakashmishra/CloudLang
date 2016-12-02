require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('task')
require('util')
require('configurations')
require('neural_net_model')

cmd = torch.CmdLine()
cmd:text()
cmd:option('--input', '', 'path to input file')
cmd:option('--output', '', 'path to output file')
cmd:option('--modelPath', '', 'path to th7 file storing a neural network')
cmd:option('--cuda', false, 'use CUDA')
cmd:text()
opt = cmd:parse(arg or {})
root_dep_label = 'ROOT'

local feature_templates = chen_manning_features['chen_manning']


local mlp, vocabs, actions
io.write("Starting up GPU ... ")
start = os.time()
require('cutorch')
require('cunn')
stop = os.time()
print(string.format("Done (%d s).", stop-start))
mlp, vocabs, actions = table.unpack(torch.load(opt.modelPath))

if opt.cuda then
    require('cutorch')
    require('cunn')
    if rand_seed then
        cutorch.manualSeed(rand_seed)
    end
    mlp:cuda()
else
    mlp:float()
end
mlp:evaluate()
print(mlp)

local conll = Format_Conll()
conll.vocabs = vocabs
local max_rows = size_of_iterator(io.lines(opt.input))+1
local ds = conll:compile_dataset(opt.input, opt.input, max_rows)
ds.tokens:narrow(2, 3, 3):zero()
local parser = GreedyParser(vocabs, actions, 
        feature_templates, mlp, opt.cuda)

print('Parsing...') 
local start_time = os.time()
local output = parser:predict_parse_tree(ds, 256)
ds.tokens:narrow(2, 3, 3):copy(output)
print(string.format('Parsing... Done (%.2f minutes).', (os.time()-start_time)/60))

conll:replace_dep_rel(opt.input, opt.output, ds)
print("Parsing results written to " .. opt.output)
