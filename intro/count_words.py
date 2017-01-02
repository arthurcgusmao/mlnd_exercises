"""Count words."""

def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    
    # TODO: Count the number of occurences of each word in s
    words_in_string = s.split(" ")
    count = {}
    for word in words_in_string:
        if(count.get(word, 0) >= 1):
            count[word] += 1
        else:
            count[word] = 1
    
    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    top = sorted(count.items(), cmp=sort_comparison)
    
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    top_n = top[:n]
    return top_n


def sort_comparison(x, y):
    """Compares the values first (numeric order) and keys second (alphabetical order), 
considering arguments are in the form [key, value]"""
    if x[1] < y[1]:
        return 1
    elif x[1] > y[1]:
        return -1
    else:
        if x[0] > y[0]:
            return 1
        elif x[0] < y[0]:
            return -1
        else:
            return 0


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)


if __name__ == '__main__':
    test_run()
