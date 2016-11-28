require 'dep.config'
require 'dep.task'
require 'data'

local function iter_paths_in_sections(from, to, dir)
    return iter_chain2(iter_map(
            function(i) 
                return find_files_recursively(
                        paths.concat(dir, string.format("%02d", i))) 
            end, iter_range(from, to)))
end

local function erase_and_concat(inp_paths, out_path)
    --print("in erase_and_concat")
    os.execute('echo "" > ' .. out_path)
    --print("erase and concat next line")
    for path in inp_paths do
	--print("erase and concat loop")
        os.execute('cat ' .. path .. ' >> ' .. out_path)   
    end
end

--
-- Compile general structures
--

local function save_info(corpus_dir, vocab_path, action_path)
    local conll = CoNLL(1, false, false)
    conll:prepare(find_files_recursively(corpus_dir), 'all', 2497203)
    print(conll.vocabs)
    torch.save(vocab_path, conll.vocabs)
    print('Input vocabulary written to ' .. vocab_path)
    
    local rels = {}
    for rel, _ in pairs(conll.vocabs.label.word2index) do
        table.insert(rels, rel)
    end
    local actions = make_actions(rels, conll.vocabs.label)
    torch.save(action_path, actions)
    print('Input vocabulary size: ' .. max_index(conll.vocabs))
    print('Action information written to ' .. action_path)
    print('Action vocabulary size: ' .. max_index(actions.vocab))
    return conll
end

-- Split data set into training, development and test sets
local data_loc = paths.concat(out_dir, 'penntree')
print(data_loc)
mkdirs(data_loc)
local penn_train_path = paths.concat(data_loc, 'train.mrg')
local penn_valid_path = paths.concat(data_loc, 'valid.mrg')
local penn_test_path = paths.concat(data_loc, 'test.mrg')
erase_and_concat(iter_paths_in_sections(2, 21, data_dir), penn_train_path)
erase_and_concat(iter_paths_in_sections(22, 22, data_dir), penn_valid_path)
erase_and_concat(iter_paths_in_sections(23, 23, data_dir), penn_test_path)

--
-- Convert to dependency using Stanford converter
--
print("Convert to dependency using Stanford converter - penn2sd")
penn2sd(data_loc, sd_dir)

-- 
-- Build standard datasets (Chen & Manning, 2014)
-- 

root_dep_label = 'root'
print("Call save_info and return conll for sd")
local conll = save_info(sd_dir, sd_vocab_path, sd_action_path)
torch.save(sd_train_path, conll:build_dataset(
        paths.concat(sd_dir, 'train.mrg.dep'), 'train', 3000000))
print('[sd] Training set written to ' .. sd_train_path)
torch.save(sd_valid_path, conll:build_dataset(
        paths.concat(sd_dir, 'valid.mrg.dep'), 'valid', 300000))
print('[sd] Development set written to ' .. sd_valid_path)
print(conll:build_dataset(
        paths.concat(sd_dir, 'test.mrg.dep'), 'test', 300000))
torch.save(sd_test_path, conll:build_dataset(
        paths.concat(sd_dir, 'test.mrg.dep'), 'test', 300000))
print('[sd] Development set written to ' .. sd_test_path)
-- os.execute('cp ' .. paths.concat(sd_dir, 'test.mrg.dep') .. ' ' .. sd_test_path)
-- print('[sd] Golden CoNLL dataset written to ' .. sd_test_path)
