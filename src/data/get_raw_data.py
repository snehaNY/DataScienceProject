import os
import logging
def extract_data(file_name,file_path):
    os.system('kaggle competitions download -c titanic -f {0} -p {1} --force'.format(file_name,file_path))

def main():
    logger = logging.getLogger(__name__)
    logger.info("getting raw data")
    train_file = 'train.csv'
    test_file = 'test.csv'
    extract_data(train_file,os.path.join(os.path.pardir,'data','raw'))
    extract_data(test_file,os.path.join(os.path.pardir,'data','raw'))
    logger.info("Raw data downloaded")
    
if __name__ == "__main__":
    #setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,format=log_fmt)
    main()
    
