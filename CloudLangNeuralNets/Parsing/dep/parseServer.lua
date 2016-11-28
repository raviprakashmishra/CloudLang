require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('data')
require('dep.task')
require('dep.config')
require('model')

local app = require('waffle')

parseText = function()
    opt = {}
    opt.input = '/home/sayak/Workspace/nndep-torch7/output.mrg.dep'
    opt.output = '/home/sayak/Workspace/nndep-torch7/output.conll'
    opt.featureType = 'chen_manning'
    opt.modelPath = '/home/sayak/Workspace/nndep-torch7/model.th7'
    opt.batchSize = 256
    opt.cuda = true
    opt.rootLabel = 'ROOT'

    root_dep_label = opt.rootLabel

    local feature_templates = predefined_feature_templates[opt.featureType]

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
end

app.get('/parse', function(req, res)
    parseText()
    res.send('Successfully parsed text')
end)
app.listen({host='127.0.0.1', port=8888})