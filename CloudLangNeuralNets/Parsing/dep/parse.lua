require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('data')
require('dep.task')
require('dep.config')
require('model')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Parse a CoNLL file')
cmd:text('Example:')
cmd:text('$> th parse.lua --id exp1 --learningRate 0.001 --max_epochs 100')
cmd:text('Options:')
cmd:option('--input', '', 'path to input file')
cmd:option('--output', '', 'path to output file')
cmd:option('--featureType', 'chen_manning', 'type of features (left or left-right)')
cmd:option('--modelPath', '', 'path to th7 file storing a neural network')
cmd:option('--batchSize', 256, 'number of sentences being parsed concurrently')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--rootLabel', 'root', 'a string for root label, different when using LTH or Stanford dependencies')
cmd:text()
opt = cmd:parse(arg or {})
assert(opt.modelPath ~= '', 'modelPath is mandatory')
root_dep_label = opt.rootLabel

assert(opt.input ~= '', 'input is mandatory')
assert(opt.output ~= '', 'ouput is mandatory')
local feature_templates = predefined_feature_templates[opt.featureType]
assert(feature_templates ~= nil, 'Unsupported feature type: ' .. opt.featureType)


local mlp, vocabs, actions
if opt.modelPath:find('%.txt$') or opt.modelPath:find('%.txt%.gz$') then
    mlp, vocabs, actions = load_stanford_model(opt.modelPath)
else
    io.write("Initializing GPU (since model may have been serialized with CUDA tensors)... ")
    start = os.time()
    require('cutorch')
    require('cunn')
    stop = os.time()
    print(string.format("Done (%d s).", stop-start))
    mlp, vocabs, actions = table.unpack(torch.load(opt.modelPath))
end
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
mlp:evaluate() -- affects dropout layer, if present
print(mlp)

local conll = CoNLL()
conll.vocabs = vocabs
local max_rows = iter_size(io.lines(opt.input))+1
local ds = conll:build_dataset(opt.input, opt.input, max_rows)
ds.tokens:narrow(2, 3, 3):zero() -- erase gold dependency tree
local parser = BatchGreedyArcStandardParser(vocabs, actions, 
        feature_templates, mlp, opt.cuda)

print('Parsing...') 
local start_time = os.time()
local output = parser:predict(ds, opt.batchSize)
ds.tokens:narrow(2, 3, 3):copy(output)
print(string.format('Parsing... Done (%.2f minutes).', (os.time()-start_time)/60))

conll:substitue_dependency(opt.input, opt.output, ds)
print("Parsing results written to " .. opt.output)
