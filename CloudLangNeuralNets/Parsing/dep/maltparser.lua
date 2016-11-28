require('dep.config')
require('paths')
require('util')

local maltparser_dir = paths.concat(out_dir, 'maltparser')
mkdirs(maltparser_dir)

local function train_and_measure(ds_name, ds_dir, algorithm)
    local train_path = paths.concat(ds_dir, 'train.mrg.dep') 
    local dev_path = paths.concat(ds_dir, 'valid.mrg.dep')
    local test_path = paths.concat(ds_dir, 'test.mrg.dep')
    local model_name = ds_name .. '_' .. algorithm
    local dev_out_path = paths.concat(maltparser_dir, model_name .. '_dev.conll')
    local test_out_path = paths.concat(maltparser_dir, model_name .. '_test.conll')

    local cmd = string.format('java -jar maltparser-1.8.1/maltparser-1.8.1.jar -c %s -w %s -a %s -i %s -m learn', 
            model_name, maltparser_dir, algorithm, train_path)
    print(cmd)
    assert(os.execute(cmd))
    
    print(string.format('*** %s, %s, dev ***', ds_name, algorithm))
    local cmd = string.format('java -jar maltparser-1.8.1/maltparser-1.8.1.jar -c %s -w %s -i %s -o %s -m parse',
            model_name, maltparser_dir, dev_path, dev_out_path)
    assert(os.execute(cmd))
    local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
            'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s',
            dev_path, dev_out_path)
    assert(os.execute(cmd))
    
    print(string.format('*** %s, %s, test ***', ds_name, algorithm))
    local cmd = string.format('java -jar maltparser-1.8.1/maltparser-1.8.1.jar -c %s -w %s -i %s -o %s -m parse',
            model_name, maltparser_dir, test_path, test_out_path)
    assert(os.execute(cmd))
    local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
            'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s',
            test_path, test_out_path)
    assert(os.execute(cmd))
end

train_and_measure('sd', sd_dir, 'stackproj')
train_and_measure('sd', sd_dir, 'nivreeager')
train_and_measure('lth', lth_dir, 'stackproj')
train_and_measure('lth', lth_dir, 'nivreeager')
