require('dep.config')
require('paths')

local function run(model_path, train_path, valid_path, test_path)
    local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
            'edu.stanford.nlp.parser.nndep.DependencyParser ' ..
            '-model %s -trainFile %s -devFile %s', model_path, train_path, valid_path)
    print(cmd)
    assert(os.execute(cmd))
    
    local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
            'edu.stanford.nlp.parser.nndep.DependencyParser ' ..
            '-model %s -testFile %s', model_path, test_path)
    print(cmd)
    assert(os.execute(cmd))
end

local sd_model_path = paths.concat(out_dir, 'sd_stanford_params.txt')
local sd_train_path = paths.concat(sd_dir, 'train.mrg.dep') 
local sd_valid_path = paths.concat(sd_dir, 'valid.mrg.dep') 
local sd_test_path = paths.concat(sd_dir, 'test.mrg.dep') 
run(sd_model_path, sd_train_path, sd_valid_path, sd_test_path)

local lth_model_path = paths.concat(out_dir, 'lth_stanford_params.txt')
local lth_train_path = paths.concat(lth_dir, 'train.mrg.dep') 
local lth_valid_path = paths.concat(lth_dir, 'valid.mrg.dep') 
local lth_test_path = paths.concat(lth_dir, 'test.mrg.dep') 
run(lth_model_path, lth_train_path, lth_valid_path, lth_test_path)
