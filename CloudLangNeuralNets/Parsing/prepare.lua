require 'configurations'
require 'task'
require 'util'

local function iterate_paths(from, to, dir)
    return chain_iterators(map_iterator(
            function(i) 
                return search_files(
                        paths.concat(dir, string.format("%02d", i))) 
            end, iterate_thru_range(from, to)))
end

local function merge_files(input, output)
    os.execute('echo "" > ' .. output)
    for path in input do
	os.execute('cat ' .. path .. ' >> ' .. output)   
    end
end

local function serialize_data(data_dir, vocab_loc, actions_loc)
    local conll = Format_Conll(1, false, false)
    print("Processing data ... ")
    conll:process_data(search_files(data_dir), 'all', 7745026)
    print('Saving input vocabulary to ' .. vocab_loc)
    torch.save(vocab_loc, conll.vocabs)
    
    local rels = {}
    for rel, _ in pairs(conll.vocabs.label.word2index) do
        table.insert(rels, rel)
    end
    local actions = take_actions(rels, conll.vocabs.label)
    print('Size of input vocabulary ' .. get_max_vocab_index(conll.vocabs))
    print('Saving correct actions to ' .. actions_loc)
    torch.save(actions_loc, actions)
    print('Size of action vocabulary ' .. get_max_vocab_index(actions.vocab))
    return conll
end

local data_loc = paths.concat(out_dir, 'treebank')
print("Creating data directories ... ")
create_directories(data_loc)
local train_data_ptb_loc = paths.concat(data_loc, 'train_model.mrg')
local valid_data_ptb_loc = paths.concat(data_loc, 'valid.mrg')
local test_data_ptb_loc = paths.concat(data_loc, 'test.mrg')
merge_files(iterate_paths(2, 21, data_dir), train_data_ptb_loc)
merge_files(iterate_paths(22, 22, data_dir), valid_data_ptb_loc)
merge_files(iterate_paths(23, 23, data_dir), test_data_ptb_loc)

print("Generating Stanford dependencies ...")
ptb_to_stanforddep(data_loc, stanford_deps_loc)

root_dep_label = 'root'
local conll = serialize_data(stanford_deps_loc, stanford_vocab_loc, stanford_actions_loc)
print('Saving training set to ' .. train_data_loc)
torch.save(train_data_loc, conll:compile_dataset(
        paths.concat(stanford_deps_loc, 'train_model.mrg.dep'), 'train_model', 5559601))
print('Saving development set to ' .. valid_data_loc)
torch.save(valid_data_loc, conll:compile_dataset(
        paths.concat(stanford_deps_loc, 'valid.mrg.dep'), 'valid', 714897))
print('Saving test set to ' .. test_data_loc)
torch.save(test_data_loc, conll:compile_dataset(
        paths.concat(stanford_deps_loc, 'test.mrg.dep'), 'test', 1470530))