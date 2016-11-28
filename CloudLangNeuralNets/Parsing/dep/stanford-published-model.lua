require('dep.config')
require('paths')

local sd_conll_test_path = paths.concat(sd_dir, 'test.mrg.dep') 

local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
        'edu.stanford.nlp.parser.nndep.DependencyParser ' ..
        '-model stanford-parser-full-2014-10-31/PTB_Stanford_params.txt.gz ' ..
        '-testFile %s', sd_conll_test_path)
print(cmd)
assert(os.execute(cmd))

local lth_conll_test_path = paths.concat(lth_dir, 'test.mrg.dep') 

local cmd = string.format('java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
        'edu.stanford.nlp.parser.nndep.DependencyParser ' ..
        '-model stanford-parser-full-2014-10-31/PTB_CoNLL_params.txt.gz ' ..
        '-testFile %s', lth_conll_test_path)
print(cmd)
assert(os.execute(cmd))