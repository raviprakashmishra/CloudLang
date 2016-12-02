require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('task')
require('util')
require('configurations')
require('neural_net_model')

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

    local feature_templates = chen_manning_features[opt.featureType]

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
    local output = parser:predict_parse_tree(ds, opt.batchSize)
    ds.tokens:narrow(2, 3, 3):copy(output)
    print(string.format('Parsing... Done (%.2f minutes).', (os.time()-start_time)/60))

    conll:replace_dep_rel(opt.input, opt.output, ds)
    print("Parsing results written to " .. opt.output)
end

app.get('/parse', function(req, res)
    parseText()
    res.send('Successfully parsed text')
end)
app.listen({host='127.0.0.1', port=8888})