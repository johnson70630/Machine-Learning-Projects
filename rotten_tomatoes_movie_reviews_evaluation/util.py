import sys


############################################################
# Milestone 3b: increment dict values 

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param scale: float
    @param dict d2: a feature vector.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key, value in d2.items():
        d1[key] = d1.get(key, 0) + scale*value
    # END_YOUR_CODE


############################################################
# Milestone 3c: dot product of 2 sparse vectors

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return sum(d1.get(key, 0)*d2[key] for key in d2)
        # END_YOUR_CODE


def readExamples(path):
    """
    Reads a set of training examples.
    """
    examples = []
    for line in open(path, "rb"):
        line = line.decode('latin-1')
        # Format of each line: <output label (+1 or -1)> <input sentence>
        y, x = line.split(' ', 1)
        examples.append((x.strip(), int(y)))
    print('Read %d examples from %s' % (len(examples), path))
    return examples


############################################################
# Milestone 5: evaluate on trainExamples and validationExamples at the end of each training epoch


def evaluatePredictor(examples, predictor):
    """
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassified examples.
    """
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)


def outputWeights(weights, path):
    """
    @param weights: dict{str: float}, holding a string of word token as key,
                                        a float of token's weight as its value.
    @param path: str, the filepath for outputting weights
    """
    print("%d weights" % len(weights))
    out = open(path, 'w', encoding='utf8')
    for f, v in sorted(list(weights.items()), key=lambda f_v: -f_v[1]):
        print('\t'.join([f, str(v)]), file=out)
    out.close()


def verbosePredict(phi, y, weights, out):
    """
    @param phi: dict[str: int], holding a string of word token as key,
                                an int of word occurrence as its value
    @param y: int, the ture label of this task
    @param weights: dict[str: float], holding a string of word token as key,
                                        a float of token's weight as its value.
    @param out: if its a str, all the answers will be outputted to out.
                On the other hand, if out is sys.stdout, all the answers will be printed on console.
    @return yy: int, prediction of either +1 or -1
    """
    yy = 1 if dotProduct(phi, weights) >= 0 else -1
    if y:
        print('Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG'), file=out)
    else:
        print('Prediction:', yy, file=out)
    for f, v in sorted(list(phi.items()), key=lambda f_v1: -f_v1[1] * weights.get(f_v1[0], 0)):
        w = weights.get(f, 0)
        print("%-30s%s * %s = %s" % (f, v, w, v * w), file=out)
    return yy


def outputErrorAnalysis(examples, featureExtractor, weights, path):
    """
    @param examples: dict[str: int], holding a string of movie review as key,
                                    its true label of int as value.
    @param featureExtractor: a function that is capable of splitting a string of movie review
                            into a dict[str: int], where str is each word token, and int is word occurrence.
    @param weights: dict[str: float], holding word token as key, the weight of each token as value.
    @param path: str, the path for outputting the overall error analysis in @param examples and @param weights
    """
    out = open(path, 'w', encoding='utf8')
    for x, y in examples:
        print('===', x, file=out)
        verbosePredict(featureExtractor(x), y, weights, out)
    out.close()


############################################################
# Milestone 5: you will incorporate the following function into your code in interactive.py


def interactivePrompt(featureExtractor, weights):
    """
    @param featureExtractor: a function capable of splitting a string of movie review into a dict[str: int],
                            where str is word token in the review, and int is its word occurrence.
    @param weights: dict[str: float], holding word token as key, the weight of each token as value.
    --------------------------------------------------
    This function uses sys.stdin.readline() to ask for user inputs. If the input is an empty,
    (empty string is considered False in Python), this function will break. Otherwise,
    the string will be fed into featureExtractor and then show the prediction on Console
    by verbosePredict.
    """
    while True:
        print('\n<<< Your review >>> ')
        x = sys.stdin.readline().strip()
        if not x: break
        phi = featureExtractor(x)
        verbosePredict(phi, None, weights, sys.stdout)
