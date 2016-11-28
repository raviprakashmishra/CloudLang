require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('data')
require('dep.task')
require('dep.feature')
require('dep.config')
require('model')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run a reimplementation of Chen & Manning (2014) experiment')
cmd:text('Example:')
cmd:text('$> th exp_chen_manning_2014.lua --cuda')
cmd:text('Options:')
cmd:option('--dependency', 'sd', 'the type of dependency, either sd (Stanford) or lth (LTH/CoNLL)')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--l1', 0, 'strength of l1 regularization')
cmd:option('--l2', 0.00000001, 'strength of l2 regularization')
cmd:option('--lrDecay', 0, 'learning rate decay factor (annealing)')
cmd:option('--dropProb', 0.5, 'Dropout probability. For each training example we randomly choose some amount of units to disable in the neural network classifier. This parameter controls the proportion of units "dropped out."')
cmd:option('--model', 'cube', 'activation function (cube or relu)')
cmd:option('--hiddenSize', '200', 'number of hidden units')
cmd:option('--batchSize', 10000, 'number of examples per batch')
cmd:option('--embeddingDims', 50, 'dimensionality of embeddings (same for words, POS and dependency labels)')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--initRange', 0.01, 'Bounds of range within which weight matrix elements should be initialized. Each element is drawn from a uniform distribution over the range [-initRange, initRange].')
cmd:option('--modelFile', 'model.th7', 'path to save model')
cmd:text()
opt = cmd:parse(arg or {})
print('Command line options: ')
print(opt)

local vocab_path, action_path, train_path, valid_path, test_path, test_conll_path
if opt.dependency == 'sd' then
    vocab_path, action_path, train_path, valid_path, test_path, test_conll_path = 
            sd_vocab_path, sd_action_path, sd_train_path, sd_valid_path, sd_test_path,
            paths.concat(sd_dir, 'test.mrg.dep')
elseif opt.dependency == 'lth' then
    root_dep_label = 'ROOT'
    vocab_path, action_path, train_path, valid_path, test_path, test_conll_path = 
            lth_vocab_path, lth_action_path, lth_train_path, lth_valid_path, lth_test_path,
            paths.concat(lth_dir, 'test.mrg.dep')
else
    error('Unknown type of dependency: ' .. opt.dependency)
end

local cfg = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.lrDecay,
    training_batch_size = opt.batchSize,
    epoch_reporting_frequency = 10,
    hidden_size = tonumber(opt.hiddenSize),
    max_epochs = opt.maxEpoch,
    l1_weight = opt.l1,
    l2_weight = opt.l2,
    mask_layer = true,
    best_path = 'output/dep/chen_manning_2014-best.th7',
    output_path = 'output/dep/chen_manning_2014-test.conll',
    model_name = 'cubenet',
    criterion = nn.ClassNLLCriterion(),
    monitoring_batch_size = 10000,
    batch_reporting_frequency = 1000,
    embedding_dims = opt.embeddingDims,
    feature_templates = ChenManningFeatures(),
    punc = false -- include punctuation in development-set evaluation
}
print('General config: ')
print(cfg)

local actions = torch.load(action_path)
local input_vocabs = torch.load(vocab_path)
if opt.model == 'cube' or opt.model == 'cubenet' then 
    cfg.mlp = new_cubenet(max_index(input_vocabs), cfg.embedding_dims, 
            cfg.feature_templates:num(), cfg.hidden_size, max_index(actions.vocab), 
            not cfg.mask_layer, opt.dropProb)
elseif opt.model == 'relu' or opt.model == 'relunet' then
    cfg.mlp = new_relunet(max_index(input_vocabs), cfg.embedding_dims, 
            cfg.feature_templates:num(), cfg.hidden_size, max_index(actions.vocab), 
            not cfg.mask_layer, opt.dropProb)
else
    error('Unsupported model: ' .. opt.model)
end
local core_mlp = cfg.mlp
if cfg.mask_layer then
    local parser = ArcStandardParser(actions)
    cfg.mlp = new_masked_net(core_mlp, parser.masks, true)
end
print('Model: ')
print(tostring(cfg.mlp))

local task = ShortestStackArcStandardOracle(cfg.feature_templates, input_vocabs, actions)
local train_ds = torch.load(train_path)
local train_x, train_y = task:build_dataset(train_ds, 'train', 2097203)
local valid_ds = torch.load(valid_path)
local valid_x, valid_y = task:build_dataset(valid_ds, 'valid', 2097203)
if not cfg.mask_layer then
    train_x = train_x[1]
    valid_x = valid_x[1]
end

if opt.cuda then
    io.write("Initializing GPU... ")
    start = os.time()
    require('cutorch')
    require('cunn')
    if rand_seed then 
        cutorch.manualSeed(rand_seed)
    end
    stop = os.time()
    print(string.format("Done (%d s).", stop-start))

    cfg.mlp:cuda()
    cfg.criterion:cuda()
    if cfg.mask_layer then
        for k = 1, #train_x do
            train_x[k] = train_x[k]:cuda()
            valid_x[k] = valid_x[k]:cuda()
        end
    else
        train_x = train_x:cuda()
        valid_x = valid_x:cuda()
    end
    train_y = train_y:cuda()
    valid_y = valid_y:cuda()
end

local parser = BatchGreedyArcStandardParser(input_vocabs, actions, 
        cfg.feature_templates, core_mlp, opt.cuda, 0)
train_parser(cfg, parser, train_x, train_y, valid_x, valid_y, valid_ds, opt.initRange)
if cfg.mask_layer then
    cfg.best_mlp = cfg.best_mlp:get_core()
end
torch.save(opt.modelFile, {cfg.best_mlp, input_vocabs, actions})

local cmd = string.format('th dep/parse.lua --rootLabel %s --modelPath %s --input %s --output %s %s',
        root_dep_label, opt.modelFile, test_conll_path, cfg.output_path, ternary(opt.cuda, '--cuda', ''))
print(cmd)
assert(os.execute(cmd))

local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
        'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s',
        test_conll_path, cfg.output_path)
print(cmd)
assert(os.execute(cmd))
