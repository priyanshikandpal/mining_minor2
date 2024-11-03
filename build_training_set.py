import nltk
from simpleDemo import extract_features
from classifier_helper import get_word_features

# Assuming process_tweet_modified is defined elsewhere
# from classifier_helper import process_tweet_modified

with open("baseline_output.txt", "r") as inpfile:
    count = 1
    tweetItems = []
    for line in inpfile:    
        count += 1
        splitArr = line.split('|')
        processed_tweet = splitArr[0].strip()
        opinion = splitArr[1].strip()
        if opinion not in ['neutral', 'negative', 'positive']:
            print(f'Error with tweet = {processed_tweet}, Line = {count}')
        tweet_item = (processed_tweet, opinion)
        tweetItems.append(tweet_item)

tweets = []    
for (words, sentiment) in tweetItems:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))
def get_words_in_tweets(tweets):
    words = []
    for tweet in tweets:
        words.extend(tweet[0])
    return words
word_features = get_word_features(get_words_in_tweets(tweets))
nltk.classify.set_word_features(word_features)
training_set = nltk.classify.apply_features(extract_features, tweets)
    
classifier = nltk.NaiveBayesClassifier.train(training_set)
tweet = 'im so sad'
print(classifier.classify(extract_features(tweet.split())))
print(nltk.classify.accuracy(classifier, training_set))
classifier.show_most_informative_features(20)
