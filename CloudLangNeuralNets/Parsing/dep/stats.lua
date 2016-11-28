require 'paths'
require 'util'
require 'data'
require 'dep.config'

function print_stats(path)
    local path_iter = ternary(paths.dirp(path), 
            find_files_recursively(path), arg_iter(path)) 
    local conll = CoNLL()
    local word_vocab = Vocabulary()
    local pos_vocab = Vocabulary()
    local label_vocab = Vocabulary()
    local sent_count = 0
    local projective_count = 0
    for path in path_iter do
        for s in conll:iter_sentences(path) do
            local heads = {0}
            for _, line in ipairs(s) do
                local fields = split(line, '\t')
                word_vocab:get_index(fields[2])
                pos_vocab:get_index(fields[4])
                label_vocab:get_index(fields[8])
                table.insert(heads, tonumber(fields[7]) + 1)
            end
            sent_count = sent_count + 1
            if is_projective(torch.LongTensor(heads)) then
                projective_count = projective_count + 1
            end
        end
    end
    print(string.format('*** %s ***', path))
    print(string.format('Sentences: %d', sent_count))
    print(string.format('Projective: %s (%.2f%%)', projective_count, 100*projective_count/sent_count))
    print(string.format('Words: %d', word_vocab.indexer:max()))
    print(string.format('Parts-of-speech: %d', pos_vocab.indexer:max()))
    print(string.format('Dependency labels: %d', label_vocab.indexer:max()))
    --print(label_vocab.word2count)
end

print_stats(sd_dir)
print_stats(paths.concat(sd_dir, 'train.mrg.dep'))
print_stats(paths.concat(sd_dir, 'valid.mrg.dep'))
print_stats(paths.concat(sd_dir, 'test.mrg.dep'))

print_stats(lth_dir)
print_stats(paths.concat(lth_dir, 'train.mrg.dep'))
print_stats(paths.concat(lth_dir, 'valid.mrg.dep'))
print_stats(paths.concat(lth_dir, 'test.mrg.dep'))
