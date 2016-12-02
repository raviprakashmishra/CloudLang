require('nn')
require('torch')
torch.setdefaulttensortype('torch.FloatTensor')

require('task')
require('util')
require('configurations')
require('neural_net_model')

cmd = torch.CmdLine()
cmd:text()
cmd:option('--cuda', false, 'CUDA or CPU')
cmd:option('--epoch', 100, 'number of epochs')
cmd:text()
opt = cmd:parse(arg or {})
print('Arguments: ')
print(opt)

local vocab_loc, actions_loc, train_data_loc, valid_data_loc, test_data_loc, test_conll_path = 
    stanford_vocab_loc, stanford_actions_loc, train_data_loc, valid_data_loc, test_data_loc, paths.concat(stanford_deps_loc, 'test.mrg.dep')

local nn_config = {
    learningRate = 0.01,
    learningRateDecay = 0,
    training_batch_size = 10000,
    epoch_reporting_frequency = 10,
    hidden_size = 200,
    max_epochs = opt.epoch,
    l1_weight = 0,
    l2_weight = 0.00000001,
    mask_layer = true,
    best_path = 'output/dep/chen_manning_2014-best.th7',
    output_path = 'output/dep/chen_manning_2014-test.conll',
    model_name = 'cubenet',
    criterion = nn.ClassNLLCriterion(),
    monitoring_batch_size = 10000,
    batch_reporting_frequency = 1000,
    embedding_dims = 50,
    feature_templates = ChenNManningFeatures(),
    punc = false
}
print('Model Configuration: ')
print(nn_config)

local actions = torch.load(actions_loc)
local input_vocabs = torch.load(vocab_loc) 
nn_config.mlp = transition_parser_neuralnet(get_max_vocab_index(input_vocabs), nn_config.embedding_dims, nn_config.feature_templates:num(), nn_config.hidden_size, 
    get_max_vocab_index(actions.vocab), not nn_config.mask_layer, 0.5)
local core_mlp = nn_config.mlp
if nn_config.mask_layer then
    local parser = TransitionParser(actions)
    nn_config.mlp = mask_neural_net(core_mlp, parser.masks, true)
end
print('Model: ')
print(tostring(nn_config.mlp))

local task = TheOracle(nn_config.feature_templates, input_vocabs, actions)
local train_ds = torch.load(train_data_loc)
local train_x, train_y = task:create_data_set(train_ds, 'train_model', 10466133)
local valid_ds = torch.load(valid_data_loc)
local valid_x, valid_y = task:create_data_set(valid_ds, 'valid', 1348198)
if not nn_config.mask_layer then
    train_x = train_x[1]
    valid_x = valid_x[1]
end

if opt.cuda then
    io.write("CUDA Enabled - Starting up GPU... ")
    start = os.time()
    require('cutorch')
    require('cunn')
    if rand_seed then 
        cutorch.manualSeed(rand_seed)
    end
    stop = os.time()
    print(string.format("Done (%d s).", stop-start))

    nn_config.mlp:cuda()
    nn_config.criterion:cuda()
    if nn_config.mask_layer then
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

local parser = GreedyParser(input_vocabs, actions, 
        nn_config.feature_templates, core_mlp, opt.cuda, 0)
train_neural_parser(nn_config, parser, train_x, train_y, valid_x, valid_y, valid_ds, 0.01)
if nn_config.mask_layer then
    nn_config.best_mlp = nn_config.best_mlp:get_core()
end
torch.save('model.th7', {nn_config.best_mlp, input_vocabs, actions})

local cmd = string.format('th parse.lua --modelPath %s --input %s --output %s %s', 
                'model.th7', test_conll_path, nn_config.output_path, ops_ternary(opt.cuda, '--cuda', ''))
print(cmd)
assert(os.execute(cmd))

local cmd = string.format('java -cp stanford-parser-full-2013-11-12/stanford-parser.jar ' ..
        'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s',
        test_conll_path, nn_config.output_path)
print(cmd)
assert(os.execute(cmd))
