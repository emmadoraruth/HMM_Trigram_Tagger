import math
import re
import sys

# Parse file to create hashes of unigram, bigram, trigram, and word-tag counts so future access will be O(1)
def hash(filename):
    unitags = {}
    bitags = {}
    tritags = {}
    wordtags = {}
    file = open(filename, 'r')
    for line in file: # Parse file by line
        parsed = str.split(line) # Parse line by whitespace
        if len(parsed) > 0:
            if parsed[1] == '1-GRAM': # Line represents unigram count
                unitags[parsed[2]] = float(parsed[0])
            elif parsed[1] == '2-GRAM': # Line represents bigram count
                bitags[(parsed[2], parsed[3])] = float(parsed[0])
            elif parsed[1] == '3-GRAM': # Line represents trigram count
                tritags[(parsed[2], parsed[3], parsed[4])] = float(parsed[0])
            else: # Line represents word-tag pair count
                wordtags[(parsed[2], parsed[3])] = float(parsed[0])
    return [wordtags, unitags, bitags, tritags]

# Calculate emission count for given word and tag
def e_val(word, tag, unitags, wordtags):
    if (tag, word) in wordtags:
        return wordtags[(tag, word)]/unitags[tag]
    else:
        return 0

# Calculate trigram estimate for given tag sequence y2, y1, y0
def q_val(y2, y1, y0, bitags, tritags):
    if (y2, y1) in bitags and (y2, y1, y0) in tritags:
        return tritags[(y2, y1, y0)]/bitags[(y2, y1)]
    else:
        return 0

# Calculate trigram estimate for each trigram sequence in file specified by filename
# Write results to file called 'trigram_estimates.txt'
def trigram_est(filename, counts_filename):
    hashes = hash(counts_filename)
    wordtags = hashes[0]
    unitags = hashes[1]
    bitags = hashes[2]
    tritags = hashes[3]
    file = open(filename, 'r')
    estimates = open('trigram_estimates.txt', 'w')
    for line in file: # Parse file by line
        parsed = str.split(line) # Parse line by whitespace
        if len(parsed) > 0:
            est = math.log(q_val(parsed[0], parsed[1], parsed[2], bitags, tritags)) # Calculate log probability of trigram estimate
            # Write output file
            estimates.write(str(parsed[0]) + ' ' + str(parsed[1]) + ' ' + str(parsed[2]) + ' ' + str(est))
        estimates.write('\n')
    file.close()
    estimates.close()

# Find maximum emission count and corresponding tag for given word
def max_e(word, unitags, wordtags):
    max_prob = 0
    prob_tag = ''
    for tag, count in unitags.iteritems(): # Iteratre over all possible tags
        if (tag, word) in wordtags:
            prob = wordtags[(tag, word)]/count # Calculate emission count probability
            if prob > max_prob: # Check if probability is maximum so far
                max_prob = prob
                prob_tag = tag
    return (max_prob, prob_tag)

# Replace words that occur fewer than 5 times in file with rare symbols
def rarify_base(filename, symbols):
    # Iterate through file and replace occurances of words in final rare dictionary with '_RARE_' symbol
    temp = abound(filename)
    abundant = temp[0]
    words = temp[1]
    file = open(filename, 'w')
    for i in range(0, len(words)):
        line = words[i]
        if len(line) > 0:
            word = line[0]
            if word not in abundant:
                for i in range(len(symbols)):
                    if symbols[i][0].search(line[0]):
                        word = symbols[i][1]
                if word == line[0]: # If rare word matches none of the special regexes, use catch-all '_RARE_' symbol
                    word = '_RARE_'
            file.write(word + ' ' + str(line[1]))
        file.write('\n')
    file.close()

# Generate list of rare word categorys
# In order of precedence (greatest to least):
# _NUM_: strings consisting of numerals, possibly with '$', '.', ',', '/', ':', and '-' characters (no more than two special characters consecutive)
# _DOT_: strings ending in '.'
# _CAPS_: strings consisting only of capital letters
# _CAP_: strings beginning with a capital letter and otherwise consisting only of lowercase letters
# _PUN_: strings consisting only of letters with at least one apostraphe or hyphen
# _NORM_: strings consisting only of lowercase letters
def rare_symbols():
    num = (re.compile(r'^([$.,/:-]?(\d)+[$.,/:-]?)+$'), '_NUM_')
    dot = (re.compile(r'^(.)+[.]$'), '_DOT_')
    caps = (re.compile(r'^[A-Z]+$'), '_CAPS_')
    cap = (re.compile(r'^[A-Z][a-z]+$'), '_CAP_')
    pun = (re.compile(r'^(([a-zA-Z]*(-|\')[a-zA-Z]+)|([a-zA-Z]+(-|\')[a-zA-Z]*))+$'), '_PUN_')
    norm = (re.compile(r'^[a-z]+$'), '_NORM_')
    return [norm, pun, cap, caps, dot, num]

# Find "abundant" (occur 5 or more times) in training file
# Returns abundant set and list of words in training file in order
def abound(filename):
    abundant = set() # Keep track of words that occur 5 or more times
    rare = {} # Keep track of words that occur fewer than 5 times with current count
    file = open(filename, 'r')
    words = []
    # Iterate through file and populate abundant and rare
    for line in file: # Parse file by line
        parsed = str.split(line) # Parse line by whitespace
        if len(parsed) > 0:
            word = parsed[0]
            if word not in abundant:
                if word not in rare:
                    rare[word] = 1
                else:
                    if rare[word] == 4:
                        abundant.add(word)
                        rare.pop(word)
                    else:
                        rare[word] += 1
        words.append(parsed)
    file.close()
    return (abundant, words)

# Finds "abundant" (occur 5 or more times) in training file from counts file generated from training file
# Any word in the counts file was abundant in training data because it was not replaced
# Excludes set of symbols used to replace rare words in the training data
def abound_counts(filename, symbols):
    abundant = set() # Keep track of words that occur 5 or more times
    file = open(filename, 'r')
    for line in file: # Parse file by line
        parsed = str.split(line) # Parse line by whitespace
        if len(parsed) > 0 and parsed[1] == 'WORDTAG' and parsed[3] not in symbols:
            abundant.add(parsed[3])
    return abundant

# Tag each word in file specified by filename with tag corresponding to highest emission count
# Write results to file called 'emission_predictions.txt'
def tagger(filename, counts_filename):
    abundant = abound_counts(counts_filename, ('_RARE'))
    hashes = hash(counts_filename)
    wordtags = hashes[0]
    unitags = hashes[1]
    bitags = hashes[2]
    tritags = hashes[3]
    file = open(filename, 'r')
    predictions = open('emission_predictions.txt', 'w')
    for line in file: # Parse file by line
        parsed = str.split(line) # Parse line by whitespace
        if len(parsed) > 0:
        # Use counts for rare words instead of word if word appears fewer than five times
            if parsed[0] in abundant:
                word = parsed[0]
            else:
                word = '_RARE_'
            max = max_e(word, unitags, wordtags) # Calculate log probability of maximum emission count
            # Write output file
            predictions.write(str(parsed[0]) + ' ' + str(max[1]) + ' ' + str(math.log(max[0])))
        predictions.write('\n')
    file.close()
    predictions.close()

# Rarify just using '_RARE_' symbol for rare words
def rarify(filename):
    rarify_base(filename, [])

# Rarify using categories for rare words
def mod_rarify(filename):
    rarify_base(filename, rare_symbols())

# Viterbi just using '_RARE_' symbol for rare words
def viterbi(filename, counts_filename):
    viterbi_base(filename, counts_filename, [])

# Viterbi using categories for rare words
def mod_viterbi(filename, counts_filename):
    viterbi_base(filename, counts_filename, rare_symbols())

# Tag each word in file specified by filename with highest probability tag calculated by Viterbi Algorithm
# Write results to file called 'viterbi_predictions.txt'
def viterbi_base(filename, counts_filename, symbols):
    abundant = abound_counts(counts_filename, [])
    hashes = hash(counts_filename)
    wordtags = hashes[0]
    unitags = hashes[1]
    bitags = hashes[2]
    tritags = hashes[3]
    file = open(filename, 'r')
    predictions = open('viterbi_predictions.txt', 'w')
    tags = unitags.keys()
    tags.append('*')
    sentence = []
    tag_seq = []
    pies = {}
    for line in file: # Parse file
        parsed = str.split(line)
        if len(parsed) > 0:
            sentence.append(parsed[0])
        else: # Blank line marks end of sentence
            n = len(sentence)
            for k in range(1, n+1): # Iterate through sentence positions
                # Use counts for rare words instead of word if word appears fewer than five times
                word = sentence[k-1]
                if sentence[k-1] not in abundant:
                    for i in range(len(symbols)):
                        if symbols[i][0].search(sentence[k-1]):
                            word = symbols[i][1]
                    if word == sentence[k-1]: # If rare word matches none of the special regexes, use catchall '_RARE_' symbol
                        word = '_RARE_'
                # Iterate through all possible bigrams of tags (u,v)
                for u in tags:
                    for v in tags:
                        max_prob = 0
                        prob_tag = ''
                        # Find tag w that maximizes liklihood of trigram (w,u,v)
                        for w in tags:
                            q = float(q_val(w, u, v, bitags, tritags)) # Trigram estimate of (w,u,v)
                            e = float(e_val(word, v, unitags, wordtags)) # Emission count of word, tag v
                            p = 0.0
                            if k == 1 and u == '*' and w == '*': # pi(0,*,*) = 1
                                p = 1.0
                            elif (k-1, w, u) in pies: # Pi value of position k-1
                                p = float(pies[(k-1, w, u)][0])
                            prob = p * q * e # Probability of tag w
                            if prob > max_prob: # Check if w is the maximum probability tag thus far
                                max_prob = prob
                                prob_tag = w
                        pies[(k, u, v)] = (max_prob, prob_tag) # Save maximum probability and corresponding tag
            max_prob = 0
            ultimate_tag = ''
            penultimate_tag = ''
            # Find most likely terminating tag bigram
            for u in tags:
                for v in tags:
                    if (n, u, v) in pies:
                        p = pies[(n, u, v)][0]
                    else:
                        p = 0
                    q = q_val(u, v, 'STOP', bitags, tritags) # Liklihood of sentence terminating in tag bigram (u,v)
                    prob = p * q
                    if prob > max_prob:
                        max_prob = prob
                        ultimate_tag = v
                        penultimate_tag = u
            # Build tag sequence in reverse
            tag_seq.append((max_prob, ultimate_tag))
            tag_seq.append((max_prob, penultimate_tag))
            for k in range(n-2, 0, -1):
                tag_seq.append(pies[(k+2, tag_seq[n-k-1][1], tag_seq[n-k-2][1])])
            # Write output file
            for k in range (n):
                predictions.write(str(sentence[k]) + ' ' + str(tag_seq[n-k-1][1]) + ' ' + str(math.log(tag_seq[n-k-1][0])) + '\n')
            predictions.write('\n')
            # Clear data structures for next sentence
            sentence = []
            pies = {}
            tag_seq = []
    predictions.close()
    file.close()

def main(args):
    if args[1] == 'rarify':
        rarify(args[2])
    elif args[1] == 'tagger':
        tagger(args[2], args[3])
    elif args[1] == 'trigram_est':
        trigram_est(args[2], args[3])
    elif args[1] == 'viterbi':
        viterbi(args[2], args[3])
    elif args[1] == 'mod_rarify':
        mod_rarify(args[2])
    elif args[1] == 'mod_viterbi':
        mod_viterbi(args[2], args[3])

if __name__ == "__main__":
    main(sys.argv)