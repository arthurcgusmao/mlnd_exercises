#------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead","could"]

def LaterWords(sample,word,distance):
    '''@param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''
    
    # TODO: Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    
    # TODO: Repeat the above process--for each distance beyond 1, evaluate the words that
    # might come after each word, and combine them weighting by relative probability
    # into an estimate of what might appear next.
    current_output = {}
    current_output[word] = 1

    for i in range(0, distance):
        current_input = current_output
        current_output = {}
        
        for w in current_input:
            cond_prob_given_w = NextWordProbability(sample, w)
            
            for poss_w in cond_prob_given_w:
                if poss_w in current_output:
                    current_output[poss_w] += cond_prob_given_w[poss_w] * current_input[w]
                else:
                    current_output[poss_w] = cond_prob_given_w[poss_w] * current_input[w]
        
        current_input = current_output

    # returning the most probable word
    most_probable = (0, '')
    for w in current_output:
        if current_output[w] > most_probable[0]:
            most_probable = (current_output[w], w)
        
    return most_probable[1]


def NextWordProbability(sampletext,word):
    text_words = sampletext.split()
    output = {}
    
    for i, w in enumerate(text_words):
        if w == word and i < len(text_words) - 1:
            next_word = text_words[i+1]
            
            if next_word in output:
                output[next_word] += 1.
            else:
                output[next_word] = 1.

    # modification: returns the probability so that they sum up to 1
    # total_occurences = 0
    # for w in output:
    #     total_occurences += output[w]
    # for w in output:
    #     output[w] /= total_occurences
    
    return output


# print LaterWords(sample_memo,"ahead",2)

test = '''
some a one, some a two, some b one,
b two, a three. Oh my gosh
'''
print LaterWords(test,"some", 2)
