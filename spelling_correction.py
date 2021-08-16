from textblob import TextBlob

words = ["Machin", "Natura", "Roos"]
for i in words:
    print(TextBlob(i).correct())