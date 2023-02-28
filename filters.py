import pandas as pd


class Filter_dataset():
    """ This class search in the annoted dataframe to return the results wanted by the user """

    def __init__(self, dataframe = None, csv_path = 'null') -> None:
        
        if csv_path=='null':
            self.dataframe = dataframe
        else:
            self.dataframe = pd.read_csv(csv_path)
        
    def filter(self, emotions, sentiments, sources, date ):
        
        if not emotions:
            filter_emotion = self.dataframe['label_emotion'].isin(emotions)
        else:
            filter_emotion = emotions

        if not sentiments != True:
            filter_sentiment = self.dataframe['label_sentiment'].isin(sentiments)
        else:
            filter_sentiment = sentiments
            
        if sources != True:
            filter_source = self.dataframe['source'].isin(sources)
        else : 
            filter_source = sources
        
        filtered_df = self.dataframe[filter_emotion & filter_sentiment & filter_source]
        
        return filtered_df