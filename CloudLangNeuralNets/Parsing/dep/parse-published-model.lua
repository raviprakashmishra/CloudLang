require 'dep.config'

local model_path = 'stanford-parser-full-2014-10-31/PTB_Stanford_params.txt.gz'
local predict_path = 'output/dep/sd_parse-published-model_test.conll'
local gold_path = paths.concat(sd_dir, 'test.mrg.dep')

cmd = string.format('th dep/parse.lua --modelPath %s --input %s --output %s', model_path, gold_path, predict_path)
print(cmd)
assert(os.execute(cmd))

cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
        'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s', gold_path, predict_path)
print(cmd)
assert(os.execute(cmd))

local model_path = 'stanford-parser-full-2014-10-31/PTB_CoNLL_params.txt.gz'
local predict_path = 'output/dep/lth_parse-published-model_test.conll'
local gold_path = paths.concat(lth_dir, 'test.mrg.dep')

cmd = string.format('th dep/parse.lua --rootLabel ROOT --modelPath %s --input %s --output %s', model_path, gold_path, predict_path)
print(cmd)
assert(os.execute(cmd))

cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
        'edu.stanford.nlp.trees.DependencyScoring -nopunc -conllx True -g %s -s %s', gold_path, predict_path)
print(cmd)
assert(os.execute(cmd))
