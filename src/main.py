from DataProcessor import *
from ModelBuilder import *
from DataExplorer import *

def main():
    #data_processor = DataProcessor()
    #data_processor.download_data(data_processor.data_with_exog)
    #model = ModelBuilder()
    #model.create_data_splits()
    explorer = DataExplorer()
    explorer.plot_hour()

if __name__ == "__main__":
    main()